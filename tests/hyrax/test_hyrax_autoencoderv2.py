import numpy as np
import pytest
import torch

from hyrax.models.hyrax_autoencoderv2 import HyraxAutoencoderV2

BATCH_SIZE = 4
# HyraxAutoencoderV2's encoder applies three stride=2 Conv2d(kernel=3) layers, and its
# decoder mirrors these with three stride=2 ConvTranspose2d layers before a CenterCrop
# trims the reconstruction back down to the original size. 20x20 comfortably survives
# the three encoder downsamples to a nonzero spatial size, matching the image size used
# in the HyraxCNN tests.
IMAGE_SHAPE = (1, 20, 20)  # (channels, width, height)
BASE_CHANNEL_SIZE = 4
LATENT_DIM = 8


def make_model_config(base_channel_size=BASE_CHANNEL_SIZE, latent_dim=LATENT_DIM, band_loss_reduction="mean"):
    """Minimal config dict containing the sections HyraxAutoencoderV2 needs from
    `hyrax_model` to build its optimizer/criterion/scheduler directly,
    without going through a full `hyrax.Hyrax()` instance."""
    return {
        "model": {
            "HyraxAutoencoderV2": {
                "base_channel_size": base_channel_size,
                "latent_dim": latent_dim,
                "final_layer": "tanh",
            }
        },
        "criterion": {"name": "torch.nn.MSELoss", "band_loss_reduction": band_loss_reduction},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01, "momentum": 0.9},
        "scheduler": {"name": None},
    }


def make_data_sample():
    """A stand-in for the `data_sample` HyraxAutoencoderV2 uses at init time to size its
    layers dynamically. Only the image tensor's shape is inspected."""
    return torch.zeros(BATCH_SIZE, *IMAGE_SHAPE)


def make_hyrax_batch(batch_size=BATCH_SIZE, seed=0):
    """Build a batch in Hyrax's collated "dict of lists" format: a single
    dict of keys where each value is a numpy array holding every datapoint
    in the batch concatenated together. This is the shape of data that
    `prepare_inputs` receives from the real data pipeline.
    """
    rng = np.random.default_rng(seed)
    images = rng.random((batch_size, *IMAGE_SHAPE)).astype(np.float32)

    return {
        "data": {
            "index": np.arange(batch_size),
            "object_id": np.array([str(i) for i in range(batch_size)]),
            "image": images,
        }
    }


def to_tensor_batch(image_array):
    """Convert the image numpy array returned by `prepare_inputs` into the
    torch tensor that `train_batch`/`validate_batch`/etc. expect. In the real
    pipeline this conversion is done by Hyrax automatically.
    """
    return torch.from_numpy(image_array)


@pytest.fixture
def model():
    """A freshly constructed HyraxAutoencoderV2 sized for 1x20x20 images."""
    return HyraxAutoencoderV2(config=make_model_config(), data_sample=make_data_sample())


def test_prepare_inputs_extracts_image():
    """`prepare_inputs` should pull `image` out of the dict-of-lists batch
    format and return it unchanged."""
    hyrax_batch = make_hyrax_batch()

    image = HyraxAutoencoderV2.prepare_inputs(hyrax_batch)

    assert image.shape == (BATCH_SIZE, *IMAGE_SHAPE)
    np.testing.assert_array_equal(image, hyrax_batch["data"]["image"])


def test_prepare_inputs_raises_without_data_key():
    """`prepare_inputs` should fail loudly if the pipeline hands it a batch
    that isn't in the expected `{"data": {...}}` shape."""
    with pytest.raises(RuntimeError, match="data"):
        HyraxAutoencoderV2.prepare_inputs({"not_data": {}})


def test_prepare_inputs_raises_without_image_key():
    """`prepare_inputs` should fail loudly if the batch's `data` dict doesn't
    contain an `image` field."""
    with pytest.raises(RuntimeError, match="image"):
        HyraxAutoencoderV2.prepare_inputs({"data": {}})


def test_forward_produces_expected_output_shape(model):
    """forward() should return one latent-space vector per batch item."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxAutoencoderV2.prepare_inputs(hyrax_batch))

    output = model(batch)

    assert output.shape == (BATCH_SIZE, LATENT_DIM)


def test_train_batch_runs_and_updates_weights(model):
    """train_batch() should return a loss and leave the model's weights changed."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxAutoencoderV2.prepare_inputs(hyrax_batch))

    weights_before = model.encoder[-1].weight.clone()
    result = model.train_batch(batch)

    assert "loss" in result
    assert isinstance(result["loss"], float)
    assert not torch.equal(weights_before, model.encoder[-1].weight)


def test_validate_batch_runs_without_updating_weights(model):
    """validate_batch() should return a loss without changing the model's weights."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxAutoencoderV2.prepare_inputs(hyrax_batch))

    weights_before = model.encoder[-1].weight.clone()
    result = model.validate_batch(batch)

    assert "loss" in result
    torch.testing.assert_close(weights_before, model.encoder[-1].weight)


def test_test_batch_runs_without_updating_weights(model):
    """test_batch() should return a loss without changing the model's weights."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxAutoencoderV2.prepare_inputs(hyrax_batch))

    weights_before = model.encoder[-1].weight.clone()
    result = model.test_batch(batch)

    assert "loss" in result
    torch.testing.assert_close(weights_before, model.encoder[-1].weight)


def test_infer_batch_returns_model_output(model):
    """infer_batch() should return one latent-space vector per batch item."""
    hyrax_batch = make_hyrax_batch()
    batch = to_tensor_batch(HyraxAutoencoderV2.prepare_inputs(hyrax_batch))

    output = model.infer_batch(batch)

    assert output.shape == (BATCH_SIZE, LATENT_DIM)


def test_train_batch_is_reproducible_with_fixed_seed():
    """Stretch goal: given the same rng seed for model initialization and the
    same input batch, training a single batch should be fully reproducible.
    """
    hyrax_batch = make_hyrax_batch(seed=42)
    batch = to_tensor_batch(HyraxAutoencoderV2.prepare_inputs(hyrax_batch))

    torch.manual_seed(1234)
    model_a = HyraxAutoencoderV2(config=make_model_config(), data_sample=make_data_sample())
    loss_a = model_a.train_batch(batch)["loss"]

    torch.manual_seed(1234)
    model_b = HyraxAutoencoderV2(config=make_model_config(), data_sample=make_data_sample())
    loss_b = model_b.train_batch(batch)["loss"]

    np.testing.assert_almost_equal(loss_a, loss_b)


@pytest.fixture(scope="function")
def hyrax_autoencoderv2_pipeline(tmp_path_factory):
    """A full `hyrax.Hyrax()` instance wired up to train/validate/test/infer
    HyraxAutoencoderV2 end-to-end against a `HyraxRandomDataset` configured to
    produce 1x20x20 single-channel images, i.e. exactly the data format
    `prepare_inputs`/`train_batch`/etc. are meant to handle.
    """
    import hyrax

    results_dir = tmp_path_factory.mktemp("hyrax_autoencoderv2_pipeline")

    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxAutoencoderV2"
    h.config["model"]["HyraxAutoencoderV2"]["base_channel_size"] = BASE_CHANNEL_SIZE
    h.config["model"]["HyraxAutoencoderV2"]["latent_dim"] = LATENT_DIM
    h.config["criterion"]["name"] = "torch.nn.MSELoss"
    h.config["train"]["epochs"] = 1
    h.config["data_loader"]["batch_size"] = BATCH_SIZE
    h.config["general"]["results_dir"] = str(results_dir)
    h.config["general"]["dev_mode"] = True

    h.config["data_request"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "test": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data")),
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path_factory.mktemp("data_infer")),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["split"] = {"train": 0.6, "validate": 0.2, "test": 0.2}
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 20
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = list(IMAGE_SHAPE)
    h.config["data_set"]["HyraxRandomDataset"]["provided_labels"] = False

    return h


def test_hyrax_autoencoderv2_train_validate_through_pipeline(hyrax_autoencoderv2_pipeline):
    """Integration test: training should run `train_batch` and
    `validate_batch` for every batch produced by the real Hyrax data
    pipeline without error."""
    h = hyrax_autoencoderv2_pipeline
    model = h.train()

    assert model is not None


def test_hyrax_autoencoderv2_test_through_pipeline(hyrax_autoencoderv2_pipeline):
    """Integration test: the `test` verb should run `test_batch` for every
    batch produced by the real Hyrax data pipeline without error, and
    return a ResultDataset."""
    from hyrax.datasets.result_dataset import ResultDataset

    h = hyrax_autoencoderv2_pipeline
    h.train()  # produce weights for the test verb to load
    result = h.test()

    assert isinstance(result, ResultDataset)


def test_hyrax_autoencoderv2_infer_through_pipeline(hyrax_autoencoderv2_pipeline):
    """Integration test: the `infer` verb should run `infer_batch` for every
    batch produced by the real Hyrax data pipeline without error."""
    h = hyrax_autoencoderv2_pipeline
    h.train()  # produce weights for the infer verb to load
    inference_results = h.infer()

    assert inference_results is not None
    assert len(inference_results) > 0
