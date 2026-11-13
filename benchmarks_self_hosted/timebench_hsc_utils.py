"""Runtime benchmark: HyraxAutoencoderV2 + HSCDataset vs. a pure-PyTorch equivalent.

This mirrors ``timebench_utils.py`` (CIFAR-10 + CNN) for the astronomy path Hyrax is built for:
an autoencoder trained on the ~1000 HSC cutouts (g/r/i FITS) published on Zenodo.

The pure-PyTorch baseline is deliberately idiomatic PyTorch: a plain ``Dataset`` that reads the
FITS files on every ``__getitem__`` (no caching) and a standalone ``nn.Module`` copy of the
``HyraxAutoencoderV2`` architecture that does not import Hyrax. The Hyrax side uses its defaults
(including ``data_set.use_cache = true``), so the comparison is "idiomatic PyTorch vs. idiomatic
Hyrax" rather than a like-for-like I/O comparison. ``use_cache`` is recorded in the results.
"""

import os
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pooch
import torch
import torch.nn as nn
import torch.optim as optim
from astropy.io import fits
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms.v2 import CenterCrop

HSC1K_EXTRACTED_DIRNAME = "hsc_8asec_1000"
HSC1K_ARCHIVE_URL = "doi:10.5281/zenodo.14498536/hsc_demo_data.zip"
HSC1K_ARCHIVE_HASH = "md5:1be05a6b49505054de441a7262a09671"

# Architecture hyperparameters. These match [model.HyraxAutoencoderV2] in hyrax_default_config.toml
# and are checked against the Hyrax model by `check_architecture_parity`.
BASE_CHANNEL_SIZE = 32
LATENT_DIM = 64

# Matches the regex HSCDataset._scan_file_names uses when no filters are configured.
HSC_FILE_REGEX = re.compile(r"([0-9]{17})_.*_(HSC-[GRIZY]).fits")


def ensure_hsc_data() -> Path:
    """Download and extract the HSC1k demo cutouts if they are not already present.

    The archive is placed in ``$HSC_DIR`` (default ``./data``) and extracted to
    ``$HSC_DIR/hsc_8asec_1000``. This is never timed.

    Returns
    -------
    Path
        Path to the directory of extracted FITS files.
    """
    data_dir = Path(os.environ.get("HSC_DIR", "./data")).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    hsc_data_dir = data_dir / HSC1K_EXTRACTED_DIRNAME

    if not hsc_data_dir.exists():
        pooch.retrieve(
            url=HSC1K_ARCHIVE_URL,
            known_hash=HSC1K_ARCHIVE_HASH,
            fname="hsc_demo_data.zip",
            path=data_dir,
            processor=pooch.Unzip(extract_dir="."),
        )

    return hsc_data_dir


def get_device() -> torch.device:
    """Pick the accelerator the same way the CIFAR benchmark does."""
    return torch.device(
        torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    )


class HSCTorchDataset(Dataset):
    """Plain PyTorch dataset over a directory of HSC cutouts downloaded by ``hyrax download``.

    It reproduces the tensors produced by Hyrax's ``HSCDataset`` with default config
    (``transform = "tanh"``, ``crop_to = false``, ``filters = false``): for each object the
    per-filter FITS images are stacked in sorted filter order, cast to float32, passed through
    ``tanh`` and center-cropped to the largest square that every image in the directory supports.

    FITS files are read on every ``__getitem__``; nothing is cached.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``<object_id>_..._<HSC-X>.fits`` files.
    """

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)

        # Group files by object id, then by filter.
        files: dict[str, dict[str, Path]] = {}
        for filepath in self.data_dir.iterdir():
            m = HSC_FILE_REGEX.match(filepath.name)
            if m is None:
                continue
            files.setdefault(m[1], {})[m[2]] = filepath

        # Mirror HSCDataset: the reference filter set is that of the first object scanned, and
        # objects that do not have exactly that set of filters are dropped.
        first_filters = sorted(next(iter(files.values())))
        self.filters = first_filters
        self.files = {
            object_id: filters for object_id, filters in files.items() if sorted(filters) == first_filters
        }
        self.object_ids = list(self.files)

        # Read every header once to find the largest square cutout all images support.
        dims = []
        for filters in self.files.values():
            for filter_name in self.filters:
                with fits.open(filters[filter_name]) as hdul:
                    dims.append(hdul[1].shape[0])
                    dims.append(hdul[1].shape[1])
        cutout = int(np.min(dims))
        self.cutout_shape = (cutout, cutout)

        self.crop = CenterCrop(size=self.cutout_shape)

    def __len__(self):
        return len(self.object_ids)

    def __getitem__(self, index):
        filters = self.files[self.object_ids[index]]
        data = [fits.getdata(filters[filter_name], memmap=False) for filter_name in self.filters]
        image = torch.from_numpy(np.array(data).astype(np.float32))
        image = torch.tanh(image)
        return self.crop(image)


def build_hsc_loader(data_dir, batch_size=128, num_workers=0, shuffle=True):
    """Build a ``DataLoader`` over the HSC cutouts in ``data_dir``.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the FITS cutouts.
    batch_size : int
        Batch size.
    num_workers : int
        Number of worker processes for data loading.
    shuffle : bool
        If True, sample with replacement (``WeightedRandomSampler`` with uniform weights), matching
        Hyrax's ``dist_data_loader``. If False, iterate sequentially.

    Returns
    -------
    DataLoader
    """
    dataset = HSCTorchDataset(data_dir)
    sampler = None
    if shuffle:
        n = len(dataset)
        weights = np.ones(n, dtype=np.float32)
        sampler = WeightedRandomSampler(weights=weights, num_samples=n, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, sampler=sampler)


def conv2d_output_size(input_size, kernel_size, padding=0, stride=1, dilation=1) -> int:
    """Output size of a Conv2d layer along one dimension (same formula as HyraxAutoencoderV2)."""
    numerator = input_size + 2 * padding - dilation * (kernel_size - 1) - 1
    return int((numerator / stride) + 1)


def conv2d_multi_layer(input_size, num_applications, **kwargs) -> int:
    """Apply `conv2d_output_size` ``num_applications`` times."""
    for _ in range(num_applications):
        input_size = conv2d_output_size(input_size, **kwargs)
    return int(input_size)


class AutoencoderV2(nn.Module):
    """Standalone copy of the ``HyraxAutoencoderV2`` architecture (no Hyrax imports).

    Parameters
    ----------
    num_input_channels : int
        Number of image channels (filters).
    image_width, image_height : int
        Spatial size of the input images.
    c_hid : int
        Base channel size of the first convolution.
    latent_dim : int
        Size of the latent vector.
    """

    def __init__(
        self,
        num_input_channels,
        image_width,
        image_height,
        c_hid=BASE_CHANNEL_SIZE,
        latent_dim=LATENT_DIM,
    ):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.image_width = image_width
        self.image_height = image_height
        self.c_hid = c_hid
        self.latent_dim = latent_dim

        self.conv_end_w = conv2d_multi_layer(image_width, 3, kernel_size=3, padding=1, stride=2)
        self.conv_end_h = conv2d_multi_layer(image_height, 3, kernel_size=3, padding=1, stride=2)

        self.encoder = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(2 * self.conv_end_h * self.conv_end_w * c_hid, latent_dim),
        )

        self.dec_linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * self.conv_end_h * self.conv_end_w * c_hid), nn.GELU()
        )
        self.final_activation = nn.Tanh()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.GELU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(
                c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            self.final_activation,
        )
        self.crop = CenterCrop(size=(image_width, image_height))

    def encode(self, x):
        """Encode a batch of images to latent vectors."""
        return self.encoder(x)

    def decode(self, z):
        """Decode a batch of latent vectors back to images."""
        x = self.dec_linear(z)
        x = x.reshape(x.shape[0], -1, self.conv_end_h, self.conv_end_w)
        x = self.decoder(x)
        return self.crop(x)

    def forward(self, x):
        """Forward pass returns the latent representation, as HyraxAutoencoderV2 does."""
        return self.encode(x)


def check_architecture_parity(sample_shape):
    """Verify that `AutoencoderV2` has the same parameters as `HyraxAutoencoderV2` built from defaults.

    Compares ``state_dict`` keys and shapes. Raises ``RuntimeError`` on any mismatch so the
    benchmark fails loudly instead of silently comparing different models. Not timed.

    Parameters
    ----------
    sample_shape : tuple
        ``(channels, height, width)`` of a single image.
    """
    from hyrax import Hyrax
    from hyrax.models.hyrax_autoencoderv2 import HyraxAutoencoderV2

    channels, height, width = sample_shape
    config = Hyrax().config
    hyrax_model = HyraxAutoencoderV2(config, data_sample=torch.zeros(1, channels, height, width))
    torch_model = AutoencoderV2(channels, height, width)

    hyrax_sd = {k: tuple(v.shape) for k, v in hyrax_model.state_dict().items()}
    torch_sd = {k: tuple(v.shape) for k, v in torch_model.state_dict().items()}
    if hyrax_sd != torch_sd:
        only_hyrax = sorted(set(hyrax_sd) - set(torch_sd))
        only_torch = sorted(set(torch_sd) - set(hyrax_sd))
        shape_diff = sorted(k for k in set(hyrax_sd) & set(torch_sd) if hyrax_sd[k] != torch_sd[k])
        raise RuntimeError(
            "AutoencoderV2 (benchmark) and HyraxAutoencoderV2 differ:\n"
            f"  only in Hyrax: {only_hyrax}\n  only in benchmark: {only_torch}\n"
            f"  shape mismatch: {shape_diff}"
        )


def benchmark_pytorch_hsc(epochs=10, batch_size=128, num_workers=0, lr=0.01, train_fraction=1.0):
    """Benchmark pure PyTorch: autoencoder training and inference on the HSC1k cutouts.

    Timing includes dataset, data loader and model construction, as in the CIFAR benchmark.
    Training weights are saved to a temporary file and reloaded for inference, since Hyrax also
    persists and reloads weights.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training and inference.
    num_workers : int
        Number of worker processes for data loading.
    lr : float
        Learning rate for SGD.
    train_fraction : float
        Recorded only; the full dataset is always used (``split = {"train": 1.0}`` on the Hyrax side).

    Returns
    -------
    dict
        Benchmark results.
    """
    data_dir = ensure_hsc_data()

    # train
    start_train = time.perf_counter()
    train_loader = build_hsc_loader(data_dir, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    dataset = train_loader.dataset
    channels = len(dataset.filters)
    width, height = dataset.cutout_shape

    net = AutoencoderV2(channels, width, height)
    device = get_device()
    net.to(device)
    print("Model running on:", next(net.parameters()).device)

    criterion = nn.MSELoss()
    # TODO: pull out momentum; as either arg or constant
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    epoch_loss = float("nan")
    for _ in range(epochs):
        # Sample-weighted epoch average, as Hyrax's _attach_epoch_metric_averaging computes it.
        # Accumulate on device and sync once per epoch.
        loss_sum = torch.zeros((), device=device)
        n_samples = 0
        for inputs in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            z = net.encode(inputs)
            x_hat = net.decode(z)
            loss = criterion(x_hat, inputs)
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach() * inputs.shape[0]
            n_samples += inputs.shape[0]
        epoch_loss = (loss_sum / n_samples).item()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        weights_path = f.name
    torch.save(net.state_dict(), weights_path)
    end_train = time.perf_counter()

    # infer
    start_infer = time.perf_counter()
    infer_loader = build_hsc_loader(data_dir, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    dataset = infer_loader.dataset
    channels = len(dataset.filters)
    width, height = dataset.cutout_shape

    net = AutoencoderV2(channels, width, height)
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.to(device)
    net.eval()

    latents = []
    with torch.no_grad():
        for inputs in infer_loader:
            inputs = inputs.to(device)
            latents.append(net.encode(inputs).cpu())
    latents = torch.cat(latents).numpy()
    end_infer = time.perf_counter()

    os.remove(weights_path)

    print(
        f"PyTorch | epochs={epochs} | batch_size={batch_size} | lr={lr:.3f} | num_workers={num_workers} | "
        f"train={end_train - start_train:.3f}s | infer={end_infer - start_infer:.3f}s | "
        f"recon_mse={epoch_loss:.6f}"
    )

    return {
        "framework": "pytorch",
        "train_fraction": train_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "use_cache": False,
        "train_time": end_train - start_train,
        "infer_time": end_infer - start_infer,
        "num_train_samples": len(train_loader.dataset),
        "num_infer_samples": len(latents),
        "recon_mse": epoch_loss,
    }


def benchmark_hyrax_hsc(epochs=10, batch_size=128, num_workers=0, lr=0.01, train_fraction=1.0):
    """Benchmark Hyrax: ``HyraxAutoencoderV2`` + ``HSCDataset`` training and inference on HSC1k.

    Parameters
    ----------
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training and inference.
    num_workers : int
        Number of worker processes for data loading.
    lr : float
        Learning rate for SGD.
    train_fraction : float
        Fraction of the dataset to train on (``split = {"train": train_fraction}``).

    Returns
    -------
    dict
        Benchmark results.
    """
    import mlflow

    from hyrax import Hyrax

    data_dir = ensure_hsc_data()

    # setup
    h = Hyrax()
    h.set_config("model.name", "HyraxAutoencoderV2")
    # The default criterion is CrossEntropyLoss, which is wrong for an autoencoder.
    h.set_config("criterion.name", "torch.nn.MSELoss")
    h.set_config("data_loader.batch_size", batch_size)
    h.set_config("data_loader.num_workers", num_workers)
    h.set_config("train.epochs", epochs)

    # Match the plain PyTorch optimizer settings so the comparison is more fair.
    # TODO: pull out momentum; as either arg or constant
    h.config["torch.optim.SGD"] = {"lr": lr, "momentum": 0.9}

    hsc_request = {
        "dataset_class": "HSCDataset",
        "data_location": str(data_dir),
        "fields": ["image"],
        "primary_id_field": "object_id",
    }
    h.set_config("data_request", {"train": {"data": dict(hsc_request)}, "infer": {"data": dict(hsc_request)}})
    h.set_config("split", {"train": train_fraction})

    use_cache = bool(h.config["data_set"]["use_cache"])

    # train
    start_train = time.perf_counter()
    h.train()
    end_train = time.perf_counter()

    # infer
    start_infer = time.perf_counter()
    inference_results = h.infer()
    end_infer = time.perf_counter()

    # Final epoch-averaged training loss, as logged by Hyrax's training loop to MLflow.
    run = mlflow.last_active_run()
    recon_mse = float(run.data.metrics["training/epoch/avg_loss"]) if run is not None else float("nan")

    print(
        f"Hyrax | epochs={epochs} | batch_size={batch_size} | lr={lr:.3f} | num_workers={num_workers} | "
        f"train={end_train - start_train:.3f}s | infer={end_infer - start_infer:.3f}s | "
        f"recon_mse={recon_mse:.6f}"
    )

    return {
        "framework": "hyrax",
        "train_fraction": train_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "use_cache": use_cache,
        "train_time": end_train - start_train,
        "infer_time": end_infer - start_infer,
        "num_infer_samples": len(inference_results),
        "recon_mse": recon_mse,
    }


def benchmark_repeated_hsc(
    framework, train_fraction=1.0, epochs=10, batch_size=128, num_workers=0, lr=0.01, repeats=3, warmup=0
):
    """Run the HSC benchmark for one framework several times and summarize the results.

    Parameters
    ----------
    framework : str
        ``"hyrax"`` or ``"pytorch"``.
    train_fraction : float
        Fraction of the dataset to train on.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training and inference.
    num_workers : int
        Number of worker processes for data loading.
    lr : float
        Learning rate.
    repeats : int
        Number of measured runs that are averaged.
    warmup : int
        Number of additional initial runs that are executed but excluded from the averages.

    Returns
    -------
    dict
        Summary with mean/std of train time, infer time and final reconstruction MSE.
    """
    if framework == "hyrax":
        runner = benchmark_hyrax_hsc
    elif framework == "pytorch":
        runner = benchmark_pytorch_hsc
    else:
        raise ValueError("framework must be 'hyrax' or 'pytorch'")

    # Untimed: make sure the data is present and the two architectures still match.
    data_dir = ensure_hsc_data()
    probe = HSCTorchDataset(data_dir)
    check_architecture_parity((len(probe.filters), *probe.cutout_shape))

    results = []
    for i in range(warmup + repeats):
        if i < warmup:
            print(f"{framework} warmup {i + 1}/{warmup}")
        else:
            print(f"{framework} repeat {i - warmup + 1}/{repeats}")
        result = runner(
            epochs=epochs,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            train_fraction=train_fraction,
        )
        if i >= warmup:
            results.append(result)

    train_times = [r["train_time"] for r in results]
    infer_times = [r["infer_time"] for r in results]
    losses = [r["recon_mse"] for r in results]

    summary = {
        "framework": framework,
        "train_fraction": train_fraction,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "lr": lr,
        "repeats": repeats,
        "warmup": warmup,
        "use_cache": results[0]["use_cache"],
        "num_samples": results[0]["num_infer_samples"],
        "train_time_mean": float(np.mean(train_times)),
        "train_time_std": float(np.std(train_times)),
        "infer_time_mean": float(np.mean(infer_times)),
        "infer_time_std": float(np.std(infer_times)),
        "recon_mse_mean": float(np.mean(losses)),
        "recon_mse_std": float(np.std(losses)),
    }

    print(
        f"{framework.upper()} avg over {repeats} runs | "
        f"train={summary['train_time_mean']:.3f}s ± {summary['train_time_std']:.3f}s | "
        f"infer={summary['infer_time_mean']:.3f}s ± {summary['infer_time_std']:.3f}s | "
        f"recon_mse={summary['recon_mse_mean']:.6f} ± {summary['recon_mse_std']:.6f}"
    )

    return summary
