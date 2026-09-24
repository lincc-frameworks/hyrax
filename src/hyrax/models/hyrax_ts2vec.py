"""TS2Vec: universal self-supervised representation learning for time series.

An implementation of TS2Vec (Yue et al., AAAI 2022, https://github.com/zhihanyue/ts2vec)
adapted to the Hyrax model interface. The reference implementation splits its logic between
a ``fit()`` method that owns the training loop and an ``encode()`` method that owns
inference; here the per-batch core of ``fit()`` lives in :meth:`HyraxTs2Vec.train_batch` and
the pooling core of ``encode()`` lives in :meth:`HyraxTs2Vec.infer_batch`, so Hyrax's own
verbs (``train``, ``train_stream``, ``infer``, ``infer_stream``) drive the loop.

The method is fully self-supervised: two overlapping random crops of the same series are
encoded, and a hierarchical contrastive loss pulls together the representations of the same
timestamp across the two views (temporal contrast) while pushing apart different instances
at the same timestamp (instance contrast). No labels are required.

Input convention
----------------
:class:`TSEncoder` consumes a single ``(batch, time, channels)`` float32 array in which
**NaN marks a missing or padded timestep**, and excludes those positions from the
convolution - so ragged light curves need no separate mask tensor once they reach the
encoder.

:func:`build_event_sequence` produces that array from ordinary padded light-curve columns,
and :meth:`HyraxTs2Vec.prepare_inputs` calls it. No per-survey dataset subclass is needed:
:class:`~hyrax.datasets.lsdb_stream_dataset.LSDBStreamDataset` already collates every nested
column into ``<column>_<subcolumn>`` plus a boolean ``<column>_<subcolumn>_mask``, and the
column names, band list, sequence length and normalization are read from
``[model.HyraxTs2Vec]``.

Encoding in ``prepare_inputs`` rather than in collation is deliberate. Hyrax's
``handle_nans`` runs over every float array produced by collation - warning when
``data_set.nan_mode`` is ``false`` and *overwriting* NaN when it is ``"zero"`` or
``"quantile"`` - and ``prepare_inputs`` is the first hook downstream of it, so it is the
earliest point at which the NaN sentinel is safe to emit.

.. warning::
    Leave ``data_set.nan_mode`` at ``false`` for this pipeline. The raw flux column reaches
    ``handle_nans`` with its own NaNs still in it, and ``"zero"`` or ``"quantile"`` would
    overwrite a missing measurement at a *valid* timestep with a number indistinguishable
    from a real one. With ``false`` those NaNs are dropped here instead, by the validity
    mask. Better still, drop them upstream in LSDB and avoid the per-batch warning.
"""

# ruff: noqa: D101, D102

import logging
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

# extra long import here to address a circular import issue
from hyrax.models.model_registry import hyrax_model

logger = logging.getLogger(__name__)


#
# Dilated convolution backbone (reference: models/dilated_conv.py)
#


class SamePadConv(nn.Module):
    """A 1D convolution padded so that the output keeps the input length."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        # An even receptive field pads one step too many on the right; trim it back off.
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out


class ConvBlock(nn.Module):
    """Two same-padded convolutions at one dilation, with a residual connection."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        # The projector makes the residual add work across a channel-count change. The final
        # block always gets one, so its residual path is learned rather than an identity.
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1) if (in_channels != out_channels or final) else None
        )

    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    """A stack of :class:`ConvBlock` whose dilation doubles at every level."""

    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(
            *[
                ConvBlock(
                    channels[i - 1] if i > 0 else in_channels,
                    channels[i],
                    kernel_size=kernel_size,
                    dilation=2**i,
                    final=(i == len(channels) - 1),
                )
                for i in range(len(channels))
            ]
        )

    def forward(self, x):
        return self.net(x)


#
# Timestamp masking (reference: models/encoder.py)
#


def generate_binomial_mask(batch_size: int, length: int, p: float = 0.5) -> torch.Tensor:
    """Return a ``(batch_size, length)`` bool mask keeping each position with probability ``p``."""
    return torch.from_numpy(np.random.binomial(1, p, size=(batch_size, length))).to(torch.bool)


def generate_continuous_mask(
    batch_size: int, length: int, n: int | float = 5, mask_length: int | float = 0.1
) -> torch.Tensor:
    """Return a ``(batch_size, length)`` bool mask with ``n`` contiguous spans masked out.

    ``n`` and ``mask_length`` may be floats, in which case they are read as fractions of
    ``length``.
    """
    res = torch.full((batch_size, length), True, dtype=torch.bool)

    if isinstance(n, float):
        n = int(n * length)
    n = max(min(n, length // 2), 1)

    if isinstance(mask_length, float):
        mask_length = int(mask_length * length)
    mask_length = max(mask_length, 1)

    for i in range(batch_size):
        for _ in range(n):
            start = np.random.randint(length - mask_length + 1)
            res[i, start : start + mask_length] = False
    return res


class TSEncoder(nn.Module):
    """Project, mask, and dilate-convolve a ``(batch, time, channels)`` series.

    NaN timesteps are detected and folded into the mask, so missing observations and padding
    contribute nothing to the convolution.
    """

    MASK_MODES = ("binomial", "continuous", "all_true", "all_false", "mask_last")

    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode="binomial"):
        super().__init__()
        if mask_mode not in TSEncoder.MASK_MODES:
            raise ValueError(f"mask_mode must be one of {TSEncoder.MASK_MODES}, got {mask_mode!r}.")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode

        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims, [hidden_dims] * depth + [output_dims], kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)

    def _build_mask(self, mask, batch_size, length, device):
        """Resolve a mask mode name into a ``(batch_size, length)`` bool tensor."""
        if mask == "binomial":
            return generate_binomial_mask(batch_size, length).to(device)
        if mask == "continuous":
            return generate_continuous_mask(batch_size, length).to(device)
        if mask == "all_true":
            return torch.full((batch_size, length), True, dtype=torch.bool, device=device)
        if mask == "all_false":
            return torch.full((batch_size, length), False, dtype=torch.bool, device=device)
        if mask == "mask_last":
            built = torch.full((batch_size, length), True, dtype=torch.bool, device=device)
            built[:, -1] = False
            return built
        raise ValueError(f"Unknown mask mode {mask!r}. Expected one of {TSEncoder.MASK_MODES}.")

    def forward(self, x, mask=None):
        """Encode ``(batch, time, input_dims)`` into ``(batch, time, output_dims)``."""
        # A timestep counts as observed only if none of its channels is NaN.
        nan_mask = ~x.isnan().any(dim=-1)
        # torch.where rather than in-place assignment, so the caller's tensor is left alone.
        x = torch.where(nan_mask.unsqueeze(-1), x, torch.zeros_like(x))

        x = self.input_fc(x)

        if mask is None:
            # Masking is a training-time augmentation only. Hyrax calls model.eval() before
            # inference, which is what makes infer_batch deterministic.
            mask = self.mask_mode if self.training else "all_true"
        if not isinstance(mask, torch.Tensor):
            mask = self._build_mask(mask, x.size(0), x.size(1), x.device)

        mask = mask & nan_mask
        x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

        x = x.transpose(1, 2)  # (batch, hidden_dims, time) for Conv1d
        x = self.repr_dropout(self.feature_extractor(x))
        return x.transpose(1, 2)  # back to (batch, time, output_dims)


#
# Hierarchical contrastive loss (reference: models/losses.py)
#


def instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Contrast each instance against the other instances at the same timestamp.

    Returns zero for a batch of one, where there are no negatives to contrast against.
    """
    batch_size = z1.size(0)
    if batch_size == 1:
        return z1.new_tensor(0.0)

    z = torch.cat([z1, z2], dim=0)  # (2B, T, C)
    z = z.transpose(0, 1)  # (T, 2B, C)
    sim = torch.matmul(z, z.transpose(1, 2))  # (T, 2B, 2B)

    # Drop the self-similarity diagonal by folding the two triangles together.
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(batch_size, device=z1.device)
    return (logits[:, i, batch_size + i - 1].mean() + logits[:, batch_size + i, i].mean()) / 2


def temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Contrast each timestamp against the other timestamps of the same instance.

    Returns zero for a single-timestamp series, where there are no negatives.
    """
    length = z1.size(1)
    if length == 1:
        return z1.new_tensor(0.0)

    z = torch.cat([z1, z2], dim=1)  # (B, 2T, C)
    sim = torch.matmul(z, z.transpose(1, 2))  # (B, 2T, 2T)

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(length, device=z1.device)
    return (logits[:, t, length + t - 1].mean() + logits[:, length + t, t].mean()) / 2


class HierarchicalContrastiveLoss(nn.Module):
    """Sum the instance and temporal contrastive losses over a max-pooling hierarchy.

    Each level halves the time axis with a max-pool, so the loss is applied from
    per-timestamp granularity all the way up to the whole series. This is an ``nn.Module``
    so that assigning it to ``self.criterion`` satisfies the ``@hyrax_model`` decorator and
    suppresses the config-driven criterion.

    Parameters
    ----------
    alpha : float
        Weight on the instance-contrastive term; ``1 - alpha`` weights the temporal term.
    temporal_unit : int
        Hierarchy level below which the temporal term is skipped.
    """

    def __init__(self, alpha: float = 0.5, temporal_unit: int = 0):
        super().__init__()
        self.alpha = alpha
        self.temporal_unit = temporal_unit

    def forward(self, z1, z2):
        alpha = self.alpha
        loss = torch.tensor(0.0, device=z1.device)
        levels = 0

        while z1.size(1) > 1:
            if alpha != 0:
                loss = loss + alpha * instance_contrastive_loss(z1, z2)
            if levels >= self.temporal_unit and (1 - alpha) != 0:
                loss = loss + (1 - alpha) * temporal_contrastive_loss(z1, z2)
            levels += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

        if z1.size(1) == 1:
            if alpha != 0:
                loss = loss + alpha * instance_contrastive_loss(z1, z2)
            levels += 1

        return loss / levels


def take_per_row(tensor: torch.Tensor, indices, num_elem: int) -> torch.Tensor:
    """Gather ``num_elem`` consecutive timesteps from each row, starting at that row's index.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape ``(batch, time, channels)``.
    indices : array-like
        Per-row start offset, length ``batch``.
    num_elem : int
        Number of consecutive timesteps to take from each row.

    Returns
    -------
    torch.Tensor
        Shape ``(batch, num_elem, channels)``.
    """
    indices = torch.as_tensor(indices, device=tensor.device, dtype=torch.long)
    offsets = torch.arange(num_elem, device=tensor.device)
    all_indices = indices[:, None] + offsets[None, :]
    rows = torch.arange(tensor.size(0), device=tensor.device)[:, None]
    return tensor[rows, all_indices]


#
# Light-curve input encoding
#

# Channel layout ahead of the one-hot band block.
FLUX_CHANNEL = 0
FLUX_ERR_CHANNEL = 1
DELTA_TIME_CHANNEL = 2
N_LEADING_CHANNELS = 3

NORMALIZE_MODES = ("median_mad", "zscore")

# 1.4826 * MAD estimates the standard deviation of normally distributed data.
MAD_TO_SIGMA = 1.4826


class EncodingSettings:
    """The ``[model.HyraxTs2Vec]`` keys that describe the input rather than the architecture.

    Resolved and validated in one place so that :func:`build_event_sequence` and
    :class:`HyraxTs2Vec` agree on the channel count, and so a bad setting is caught during
    model pre-flight rather than midway through a run.

    ``time_field``, ``flux_field`` and ``flux_err_field`` are flattened column names, as
    ``LSDBStreamDataset`` exposes nested columns. ``band_field`` is ``None`` for a
    single-band survey, where the one-hot block is dropped entirely and ``bands`` is empty;
    otherwise ``bands`` gives the one-hot channel order. ``max_sequence_length`` is how many
    observations are kept per object, which fixes the width of the time axis, and
    ``normalize`` is one of :data:`NORMALIZE_MODES` or ``False``. ``n_channels`` is
    ``3 + len(bands)``, the model's input width.

    Parameters
    ----------
    config : dict
        The runtime configuration.
    """

    def __init__(self, config: dict):
        model_config = config["model"]["HyraxTs2Vec"]

        self.time_field = str(model_config["time_field"])
        self.flux_field = str(model_config["flux_field"])
        self.flux_err_field = str(model_config["flux_err_field"])

        # `false` means a single-band survey (TESS, Kepler): there is no band column to
        # read, so the one-hot block is dropped and a series is just flux, flux error, and
        # time gap.
        band_field = model_config["band_field"]
        self.band_field = None if band_field is False else str(band_field)

        self.light_curve_fields = tuple(
            field
            for field in (self.time_field, self.flux_field, self.flux_err_field, self.band_field)
            if field is not None
        )

        self.bands = [] if self.band_field is None else [str(band) for band in model_config["bands"]]
        if self.band_field is not None and not self.bands:
            raise ValueError(
                "config['model']['HyraxTs2Vec']['bands'] must list at least one band; it "
                "sets the width of the one-hot band channels. Set band_field = false "
                "instead if the catalog is single-band."
            )
        self.band_index = {band: index for index, band in enumerate(self.bands)}

        max_sequence_length = model_config["max_sequence_length"]
        # `false` is the TOML "not set" sentinel and bool is a subclass of int, so
        # int(False) would silently become a zero-length sequence here.
        if max_sequence_length is False or int(max_sequence_length) < 1:
            raise ValueError(
                "config['model']['HyraxTs2Vec']['max_sequence_length'] must be a positive "
                f"integer, got {max_sequence_length!r}."
            )
        self.max_sequence_length = int(max_sequence_length)

        self.normalize = model_config["normalize"]
        if self.normalize is not False and self.normalize not in NORMALIZE_MODES:
            raise ValueError(
                "config['model']['HyraxTs2Vec']['normalize'] must be one of "
                f"{NORMALIZE_MODES}, or false to disable it. Got {self.normalize!r}."
            )

        self.n_channels = N_LEADING_CHANNELS + len(self.bands)


def _select_columns(data: dict, settings: EncodingSettings):
    """Pull the light-curve columns and the validity mask out of a collated batch.

    Returns ``(time, flux, flux_err, band, valid)``, each ``(batch, length)``. ``band`` is
    ``None`` for a single-band catalog. ``valid`` starts from the collation mask and then
    drops non-finite observations, so it is the single source of truth for what is real.
    """
    missing = [field for field in settings.light_curve_fields if field not in data]
    if missing:
        raise RuntimeError(
            f"HyraxTs2Vec could not find the field(s) {missing} in the collated batch. "
            f"Available fields: {sorted(data)}. Nested columns are exposed as "
            "'<column>_<subcolumn>', so set time_field / flux_field / flux_err_field / "
            "band_field in [model.HyraxTs2Vec] to those flattened names, and list them in "
            "the data request's 'fields'."
        )

    # Upcast for the arithmetic below. This does not recover precision already lost: the
    # streaming provider casts float columns to float32 before collation, which leaves an
    # MJD timestamp good to roughly six minutes. That is fine for the gap channel on a
    # survey with nightly cadence, and worth knowing about for one with finer sampling.
    time = np.asarray(data[settings.time_field], dtype=np.float64)
    flux = np.asarray(data[settings.flux_field], dtype=np.float64)
    flux_err = np.asarray(data[settings.flux_err_field], dtype=np.float64)
    band = None if settings.band_field is None else np.asarray(data[settings.band_field])

    # A fixed-length nested column is stacked directly and has no mask; everything in it
    # is then a real observation.
    mask = data.get(f"{settings.time_field}_mask")
    valid = np.ones(time.shape, dtype=bool) if mask is None else np.asarray(mask).astype(bool)

    shapes = {array.shape for array in (time, flux, flux_err, valid)}
    if band is not None:
        shapes.add(band.shape)
    if len(shapes) > 1:
        raise RuntimeError(
            f"HyraxTs2Vec expects the light-curve columns to collate to a common shape, "
            f"but got {sorted(shapes)}. Every column must come from the same nested "
            "column, so that one padded row describes one object's observations."
        )

    # Drop non-finite observations rather than carrying NaN into the series: the mask is
    # the single source of truth, and the encoder reads NaN as "padding".
    valid = valid & np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)

    return time, flux, flux_err, band, valid


def _sort_and_trim(columns, valid, length):
    """Sort each row by time and fit it to exactly ``length`` timesteps.

    Invalid entries sort to the tail, so the kept observations are the earliest real ones,
    matching what a per-object sort-then-truncate would give.
    """
    time = columns[0]

    # +inf as the sort key pushes padding and non-finite observations past everything real.
    key = np.where(valid, time, np.inf)
    order = np.argsort(key, axis=1, kind="stable")

    columns = [np.take_along_axis(column, order, axis=1) for column in columns]
    valid = np.take_along_axis(valid, order, axis=1)

    current = valid.shape[1]
    if current > length:
        columns = [column[:, :length] for column in columns]
        valid = valid[:, :length]
    elif current < length:
        pad = ((0, 0), (0, length - current))
        columns = [np.pad(column, pad) for column in columns]
        valid = np.pad(valid, pad, constant_values=False)

    return columns, valid


def _normalize_flux(flux, flux_err, valid, mode):
    """Center and rescale each object's flux, applying the same scale to its error.

    Per-object rather than per-batch: raw fluxes span orders of magnitude across bands and
    objects, and TS2Vec applies no normalization of its own. The shared scale is what keeps
    the error interpretable next to the flux.
    """
    if mode is False:
        return flux, flux_err

    observed = np.where(valid, flux, np.nan)

    # An object with no valid observations makes every reduction below an all-NaN slice.
    # Those rows are masked off entirely at the end, so warning about them is just noise.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "median_mad":
            center = np.nanmedian(observed, axis=1, keepdims=True)
            scale = MAD_TO_SIGMA * np.nanmedian(np.abs(observed - center), axis=1, keepdims=True)
            # A MAD of zero means over half the points are identical; fall back to std.
            scale = np.where(scale > 0.0, scale, np.nanstd(observed, axis=1, keepdims=True))
        else:  # "zscore", validated by EncodingSettings
            center = np.nanmean(observed, axis=1, keepdims=True)
            scale = np.nanstd(observed, axis=1, keepdims=True)

    # A constant light curve: centering still means something, rescaling does not. This
    # also catches the NaN scale of an all-invalid row.
    scale = np.where(scale > 0.0, scale, 1.0)
    center = np.where(np.isfinite(center), center, 0.0)

    return (flux - center) / scale, flux_err / scale


def _band_indices(band, valid, settings: EncodingSettings):
    """Map band values to one-hot column offsets, returning ``-1`` for unrecognized ones.

    Accepts integer band codes directly and matches everything else against the configured
    band names. Mapping goes through ``np.unique`` so the dict lookups cost one per distinct
    band rather than one per observation. Unrecognized bands are reported only where the
    mask says there is a real observation, so zero padding is not mistaken for a band.
    """
    if np.issubdtype(band.dtype, np.integer):
        codes = band.astype(np.int64)
        known = (codes >= 0) & (codes < len(settings.bands))
        unknown = {int(code) for code in np.unique(codes[valid & ~known])}
        return np.where(known, codes, -1), unknown

    keys = np.char.decode(band) if band.dtype.kind == "S" else band.astype(str)
    keys = np.char.strip(keys)

    uniques, inverse = np.unique(keys, return_inverse=True)
    mapped = np.array([settings.band_index.get(str(u), -1) for u in uniques], dtype=np.int64)
    # numpy < 2 flattens `inverse`; reshaping is a no-op on numpy >= 2.
    codes = mapped[inverse.reshape(keys.shape)]

    unknown = {str(key) for key in np.unique(keys[valid & (codes < 0)])}
    return codes, unknown


def build_event_sequence(data_dict: dict, config: dict) -> np.ndarray:
    """Turn padded light-curve columns into the event sequence :class:`TSEncoder` consumes.

    Observations are sorted by time and laid out as an ordered sequence of events, one
    timestep per observation, with channels::

        0                      flux, normalized per object
        1                      flux error, on the same scale as the flux
        2                      log1p of the gap since the previous observation (0 for the first)
        3 .. 3 + len(bands)    one-hot band indicator

    Uneven sampling is therefore carried as a *feature* (channel 2) rather than being
    resampled away, and a light curve that spans several filters stays a single sequence.
    Cross-matching more surveys to add bands only widens the one-hot block; the encoder
    sizes its input layer from ``bands``, so no architecture change is needed.

    The input columns are whatever the dataset collated under the names configured in
    ``[model.HyraxTs2Vec]``. ``LSDBStreamDataset`` produces them for every nested column
    without any per-survey subclass: ``<column>_<subcolumn>`` padded to the batch's longest
    row, plus a boolean ``<column>_<subcolumn>_mask``.

    Parameters
    ----------
    data_dict : dict
        The collated batch, with a ``"data"`` key holding the light-curve columns.
    config : dict
        The runtime configuration.

    Returns
    -------
    numpy.ndarray
        ``(batch, max_sequence_length, 3 + len(bands))`` float32, NaN at every padded or
        missing timestep.
    """
    if "data" not in data_dict:
        raise RuntimeError(
            "HyraxTs2Vec could not find a 'data' key in the collated batch. Name the "
            "dataset in your [data_request] group 'data', e.g. "
            "{'train_stream': {'data': {...}}}."
        )

    settings = EncodingSettings(config)
    time, flux, flux_err, band, valid = _select_columns(data_dict["data"], settings)

    columns = [time, flux, flux_err] if band is None else [time, flux, flux_err, band]
    columns, valid = _sort_and_trim(columns, valid, settings.max_sequence_length)
    time, flux, flux_err = columns[0], columns[1], columns[2]
    band = columns[3] if band is not None else None

    flux, flux_err = _normalize_flux(flux, flux_err, valid, settings.normalize)

    # Log-compressed because season-length gaps otherwise dwarf intra-night cadence. The
    # first observation of each object has no predecessor, and neither does one whose
    # predecessor was dropped, so both get a gap of zero.
    gaps = np.zeros(time.shape, dtype=np.float64)
    gaps[:, 1:] = np.where(valid[:, :-1], np.diff(time, axis=1), 0.0)
    gaps = np.log1p(np.clip(gaps, 0.0, None))

    series = np.zeros((*time.shape, settings.n_channels), dtype=np.float32)
    series[..., FLUX_CHANNEL] = flux
    series[..., FLUX_ERR_CHANNEL] = flux_err
    series[..., DELTA_TIME_CHANNEL] = gaps

    if band is not None:
        codes, unknown = _band_indices(band, valid, settings)
        if unknown:
            logger.warning(
                f"Ignoring observations in unrecognized band(s) {sorted(unknown)}; they get "
                f"an all-zero band indicator. Configured bands: {settings.bands}. Add them "
                "to config['model']['HyraxTs2Vec']['bands'] to encode them."
            )
        known = valid & (codes >= 0)
        rows, timesteps = np.nonzero(known)
        series[rows, timesteps, N_LEADING_CHANNELS + codes[known]] = 1.0

    # NaN marks padding for the encoder. Applying it here rather than during collation is
    # deliberate: Hyrax's handle_nans runs over every collated float array and would warn
    # about the sentinel (data_set.nan_mode = false) or overwrite it ("zero"/"quantile").
    return np.where(valid[..., None], series, np.float32("nan"))


#
# The Hyrax model (reference: ts2vec.py)
#


@hyrax_model
class HyraxTs2Vec(nn.Module):
    """TS2Vec, a self-supervised encoder for multivariate time series.

    Consumes ``(batch, time, channels)`` float32 input where NaN marks a missing or padded
    timestep, and produces one fixed-length representation per object from
    :meth:`infer_batch` - ready for ``reduce_dimensions`` and the ``visualize`` verbs.

    ``channels`` is the event-sequence encoding built by :func:`build_event_sequence`: flux,
    flux error, log-scaled time gap, then a one-hot band indicator. Adding bands (by
    cross-matching more surveys) only widens ``channels``; no architecture change is needed.

    Notes
    -----
    This model sets ``self.optimizer`` itself (AdamW at ``[model.HyraxTs2Vec].learning_rate``)
    rather than taking the config-driven optimizer, because the Hyrax default is SGD at
    ``lr = 0.01`` while TS2Vec is specified on AdamW at ``1e-3``. That is a supported path,
    but it does mean Hyrax logs one benign "Both model and config define an optimizer"
    warning per run.

    Differences from the reference implementation, both deliberate:

    - No SWA. The reference wraps the encoder in ``torch.optim.swa_utils.AveragedModel`` and
      encodes from the averaged weights. Skipped here: it puts non-gradient parameters into
      the optimizer, complicates ``state_dict`` save/load, and risks DDP unused-parameter
      errors.
    - No ``max_train_length`` splitting. The reference chops over-long series into segments
      stacked along the batch axis before training. Sequence length is instead bounded by the
      dataset's ``max_sequence_length``.
    """

    def __init__(self, config, data_sample=None):
        super().__init__()
        # Must be set before __init__ returns: the @hyrax_model decorator reads self.config
        # immediately afterwards to build the criterion, optimizer, and scheduler.
        self.config = config

        model_config = config["model"]["HyraxTs2Vec"]

        if data_sample is None:
            raise RuntimeError(
                "HyraxTs2Vec needs a data_sample to size its input layer. Hyrax normally "
                "supplies one from your dataset; when constructing the model by hand, pass "
                "a (batch, time, channels) array."
            )
        if getattr(data_sample, "ndim", None) != 3:
            raise RuntimeError(
                "HyraxTs2Vec expects a 3-dimensional (batch, time, channels) data sample, but "
                f"got shape {getattr(data_sample, 'shape', type(data_sample))}. Hyrax builds "
                "it with build_event_sequence(); pass an array of that shape when "
                "constructing the model by hand."
            )

        self.input_dims = int(data_sample.shape[-1])
        logger.debug(
            f"Found shape: {tuple(data_sample.shape)} in data sample, "
            f"using input_dims={self.input_dims} to initialize model."
        )

        self.output_dims = model_config["output_dims"]
        self.hidden_dims = model_config["hidden_dims"]
        self.depth = model_config["depth"]
        self.mask_mode = model_config["mask_mode"]
        self.temporal_unit = model_config["temporal_unit"]
        self.encoding_window = model_config["encoding_window"]

        self.encoder = TSEncoder(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            mask_mode=self.mask_mode,
        )

        self.criterion = HierarchicalContrastiveLoss(
            alpha=model_config["alpha"], temporal_unit=self.temporal_unit
        )
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=model_config["learning_rate"])

    def forward(self, x, mask=None):
        """Encode a batch of series into per-timestamp representations.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, time, channels)``, NaN at missing or padded timesteps.
        mask : str or torch.Tensor, optional
            Overrides the configured masking strategy for this call.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, time, output_dims)``.
        """
        return self.encoder(x, mask)

    def _drop_empty_rows(self, x):
        """Drop rows that are NaN everywhere, as the reference's ``fit`` does.

        Safe during training because the loss does not need to align with ``object_id``.
        """
        all_nan = torch.isnan(x).flatten(start_dim=1).all(dim=1)
        if bool(all_nan.any()):
            logger.debug(f"Dropping {int(all_nan.sum())} all-NaN series from the batch.")
            x = x[~all_nan]
        return x

    def _contrastive_loss(self, x):
        """Encode two overlapping random crops of ``x`` and return their contrastive loss."""
        min_length = 2 ** (self.temporal_unit + 1)
        ts_l = x.size(1)
        if ts_l < min_length:
            raise RuntimeError(
                f"HyraxTs2Vec needs sequences of at least {min_length} timesteps for "
                f"temporal_unit={self.temporal_unit}, but got {ts_l}. Either raise "
                "config['model']['HyraxTs2Vec']['max_sequence_length'] or lower "
                "config['model']['HyraxTs2Vec']['temporal_unit']."
            )

        # Two crops that overlap on a shared window of crop_l timesteps, each seeing a
        # different amount of surrounding context. The shared window is what the loss
        # contrasts; the differing context is the augmentation.
        crop_l = np.random.randint(low=min_length, high=ts_l + 1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

        # self.forward rather than self.encoder so a DDP-wrapped self still routes through
        # the wrapper and synchronizes gradients.
        out1 = self.forward(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
        out1 = out1[:, -crop_l:]

        out2 = self.forward(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
        out2 = out2[:, :crop_l]

        return self.criterion(out1, out2)

    def train_batch(self, batch):
        """Run one self-supervised training step, the inner loop of the reference ``fit()``.

        Parameters
        ----------
        batch : torch.Tensor
            Shape ``(batch, time, channels)``, NaN at missing or padded timesteps.

        Returns
        -------
        dict
            ``{"loss": float}`` for the current batch.
        """
        x = self._drop_empty_rows(batch)
        if x.size(0) == 0:
            logger.warning("Every series in this batch was entirely missing; skipping the step.")
            return {"loss": 0.0}

        loss = self._contrastive_loss(x)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validate_batch(self, batch):
        """Compute the training loss without updating any weights.

        Parameters
        ----------
        batch : torch.Tensor
            Shape ``(batch, time, channels)``, NaN at missing or padded timesteps.

        Returns
        -------
        dict
            ``{"loss": float}`` for the current batch.
        """
        x = self._drop_empty_rows(batch)
        if x.size(0) == 0:
            logger.warning("Every series in this batch was entirely missing; reporting zero loss.")
            return {"loss": 0.0}

        return {"loss": self._contrastive_loss(x).item()}

    def test_batch(self, batch):
        """Identical to :meth:`validate_batch`.

        Parameters
        ----------
        batch : torch.Tensor
            Shape ``(batch, time, channels)``, NaN at missing or padded timesteps.

        Returns
        -------
        dict
            ``{"loss": float}`` for the current batch.
        """
        return self.validate_batch(batch)

    def infer_batch(self, batch):
        """Produce the latent representation of each series, the core of ``encode()``.

        The model is in eval mode here, so :class:`TSEncoder` resolves its mask to
        ``"all_true"`` and the output is deterministic.

        Series that are entirely missing are **not** dropped, because the inference output
        must stay aligned one-to-one with ``object_id``; they encode a fully masked,
        all-zero input.

        Parameters
        ----------
        batch : torch.Tensor
            Shape ``(batch, time, channels)``, NaN at missing or padded timesteps.

        Returns
        -------
        torch.Tensor
            ``(batch, output_dims)`` when ``encoding_window`` is ``"full_series"``, or
            ``(batch, time, output_dims * levels)`` when it is ``"multiscale"``.
        """
        out = self.forward(batch)

        if self.encoding_window == "full_series":
            # Max-pool the whole time axis down to one vector per object.
            pooled = F.max_pool1d(out.transpose(1, 2), kernel_size=out.size(1))
            return pooled.squeeze(-1)

        if self.encoding_window == "multiscale":
            representations = []
            p = 0
            while (1 << p) + 1 < out.size(1):
                pooled = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size=(1 << (p + 1)) + 1,
                    stride=1,
                    padding=1 << p,
                )
                representations.append(pooled.transpose(1, 2))
                p += 1
            return torch.cat(representations, dim=-1)

        raise ValueError(
            "config['model']['HyraxTs2Vec']['encoding_window'] must be 'full_series' or "
            f"'multiscale', got {self.encoding_window!r}."
        )

    @staticmethod
    def prepare_inputs(data_dict):
        """Build the event sequence this model consumes from a collated batch.

        The real work is in :func:`build_event_sequence`; this is a shim so that the
        function written out to ``prepare_inputs.py`` next to the saved weights stays short
        and self-contained. That does mean a checkpoint pins the *settings* of the encoding
        - they are saved in the run's ``runtime_config.toml`` - rather than its code, which
        tracks the installed Hyrax.

        Parameters
        ----------
        data_dict : dict
            The collated batch, with a ``"data"`` key holding the padded light-curve
            columns named in ``[model.HyraxTs2Vec]``.

        Returns
        -------
        numpy.ndarray
            Shape ``(batch, max_sequence_length, channels)``, float32, NaN at padded
            timesteps. Hyrax converts this to a tensor and moves it to the right device.
        """
        # This function's source is written out to prepare_inputs.py next to the saved
        # weights and re-executed at load time, so it must import what it needs itself.
        from hyrax import get_context
        from hyrax.models.hyrax_ts2vec import build_event_sequence

        return build_event_sequence(data_dict, get_context()["config"])
