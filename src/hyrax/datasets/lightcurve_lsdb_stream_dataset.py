"""Streaming dataset that turns ragged multi-band light curves into a fixed-width array.

:class:`LightCurveLSDBStreamDataset` extends
:class:`~hyrax.datasets.lsdb_stream_dataset.LSDBStreamDataset` with the collation a
sequence model needs: it reads four nested light-curve columns (time, flux, flux error, and
band) and emits one ``(batch, time, channels)`` array per batch.

**Event-sequence encoding.** Observations are sorted by time and laid out as an ordered
sequence of events, one timestep per observation, with channels::

    0                      flux, normalized per object
    1                      flux error, on the same scale as the flux
    2                      log1p of the gap since the previous observation (0 for the first)
    3 .. 3 + len(bands)    one-hot band indicator

Uneven sampling is therefore carried as a *feature* (channel 2) rather than being resampled
away, and a light curve that spans several filters stays a single sequence. Cross-matching
more surveys to add bands only widens the one-hot block - the consuming model's architecture
adapts on its own, because it reads its input width from a data sample.

Single-band surveys (TESS, Kepler) have no band column to read. Set ``band_field = false``
for those: the one-hot block is dropped and a series is just the three leading channels.

**Padding is zeros plus a mask, never NaN.** Hyrax's ``handle_nans`` runs over every float
array a collate function produces: with ``data_set.nan_mode = false`` it warns on every
batch, and with ``"zero"`` or ``"quantile"`` it silently *overwrites* NaN. So this dataset
emits zero padding and an explicit ``series_mask``, and
:meth:`~hyrax.models.hyrax_ts2vec.HyraxTs2Vec.prepare_inputs` converts the mask to NaN once
the batch is past that hook. Non-finite observations are dropped rather than passed through,
so ``series`` is always NaN-free and the mask is the single source of truth.

.. note::
    This class is named for what it does rather than for one survey: the default column names
    and band list are LSST's, but nothing here is LSST-specific, which is what lets a
    cross-matched catalog spanning several surveys use it unchanged.

.. code-block:: python

    import hyrax
    import lsdb
    from hyrax.datasets import LSDBStreamDataset

    catalog = lsdb.open_catalog("...")
    # Register the catalog rather than passing a path: the parent class forwards flattened
    # field names ("lightcurve_psfFlux") to lsdb.open_catalog(columns=...), which will not
    # match a real HATS nested column. An "lsdb://" location skips that path entirely.
    data_location = LSDBStreamDataset.register_catalog("lsst_lightcurves", catalog)

    hy = hyrax.Hyrax()
    hy.config["model"]["name"] = "HyraxTs2Vec"
    hy.config["data_request"] = {
        "train_stream": {
            # The group must be named "data" - HyraxTs2Vec.prepare_inputs reads that key.
            "data": {
                "dataset_class": "LightCurveLSDBStreamDataset",
                "data_location": data_location,
                "primary_id_field": "objectId",
                # Always list `fields` explicitly. Omitting it makes every column a model
                # input, including string and ragged ones.
                "fields": [
                    "lightcurve_midpointMjdTai",
                    "lightcurve_psfFlux",
                    "lightcurve_psfFluxErr",
                    "lightcurve_band",
                ],
            }
        }
    }
"""

import logging

import numpy as np

from hyrax.datasets.lsdb_stream_dataset import LSDBStreamDataset

logger = logging.getLogger(__name__)

# Channel layout ahead of the one-hot band block.
FLUX_CHANNEL = 0
FLUX_ERR_CHANNEL = 1
DELTA_TIME_CHANNEL = 2
N_LEADING_CHANNELS = 3

NORMALIZE_MODES = ("median_mad", "zscore")

# 1.4826 * MAD estimates the standard deviation of normally distributed data.
MAD_TO_SIGMA = 1.4826


class LightCurveLSDBStreamDataset(LSDBStreamDataset):
    """Stream a HATS catalog and collate its nested light curves into an event sequence.

    Configured in ``[data_set.LightCurveLSDBStreamDataset]``; the stream itself
    (``stream_type``, ``partitions_per_chunk``, ``shuffle``, ``seed``) is configured in
    ``[data_set.LSDBStreamDataset]``, which the parent reads by literal class name so
    subclasses share one block.

    Emits three fields per batch:

    ``series``
        ``(batch, max_sequence_length, 3 + len(bands))`` float32, zero-padded.
    ``series_mask``
        ``(batch, max_sequence_length)`` int64, 1 for a real observation and 0 for padding.
    ``series_lengths``
        ``(batch,)`` int64, the number of real observations, clamped to
        ``max_sequence_length`` so it always agrees with the mask.

    Any other requested field that collates to a uniform numeric array is passed through.
    """

    def __init__(self, config: dict, data_location=None):
        super().__init__(config, data_location)
        self._init_collation(config)

    def _init_collation(self, config: dict) -> None:
        """Read and validate the collation settings.

        Split out from ``__init__`` so the collation can be configured - and tested - without
        opening a catalog, which the parent's ``__init__`` requires.
        """
        # Looked up by literal name, matching the parent, so a further subclass inherits
        # these defaults instead of needing a config block of its own.
        ds_config = config["data_set"]["LightCurveLSDBStreamDataset"]

        self.time_field = str(ds_config["time_field"])
        self.flux_field = str(ds_config["flux_field"])
        self.flux_err_field = str(ds_config["flux_err_field"])

        # `false` means a single-band survey (TESS, Kepler): there is no band column to read,
        # so the one-hot block is dropped entirely and a series is just flux, flux error, and
        # time gap. The model sizes its input layer from a data sample, so it adapts on its own.
        band_field = ds_config["band_field"]
        self.band_field = None if band_field is False else str(band_field)

        self.light_curve_fields = tuple(
            field
            for field in (self.time_field, self.flux_field, self.flux_err_field, self.band_field)
            if field is not None
        )

        self.bands = [] if self.band_field is None else [str(band) for band in ds_config["bands"]]
        if self.band_field is not None and not self.bands:
            raise ValueError(
                "config['data_set']['LightCurveLSDBStreamDataset']['bands'] must list at "
                "least one band; it sets the width of the one-hot band channels. Set "
                "band_field = false instead if the catalog is single-band."
            )
        self._band_index = {band: index for index, band in enumerate(self.bands)}

        max_sequence_length = ds_config["max_sequence_length"]
        # `false` is the TOML "not set" sentinel and bool is a subclass of int, so int(False)
        # would silently become a zero-length sequence here.
        if max_sequence_length is False or int(max_sequence_length) < 1:
            raise ValueError(
                "config['data_set']['LightCurveLSDBStreamDataset']['max_sequence_length'] "
                f"must be a positive integer, got {max_sequence_length!r}."
            )
        self.max_sequence_length = int(max_sequence_length)

        self.normalize = ds_config["normalize"]
        if self.normalize is not False and self.normalize not in NORMALIZE_MODES:
            raise ValueError(
                "config['data_set']['LightCurveLSDBStreamDataset']['normalize'] must be one "
                f"of {NORMALIZE_MODES}, or false to disable it. Got {self.normalize!r}."
            )

        self.n_channels = N_LEADING_CHANNELS + len(self.bands)

    #
    # Collation
    #

    def collate(self, samples: list[dict]) -> dict:
        """Build one ``(batch, time, channels)`` event sequence from a list of samples.

        A dataset-level ``collate`` rather than per-field ``collate_<field>`` hooks, because
        the encoding has to see time, flux, flux error, and band together. Defining it also
        keeps the default field collation away from the band column, whose ragged string
        arrays it cannot stack.

        Parameters
        ----------
        samples : list of dict
            One dict per object, mapping requested field name to that object's value. Light
            curve fields hold variable-length 1-D arrays.

        Returns
        -------
        dict
            ``series``, ``series_mask``, ``series_lengths``, plus any passed-through fields.
        """
        self._check_fields(samples[0])

        n_samples = len(samples)
        length = self.max_sequence_length

        series = np.zeros((n_samples, length, self.n_channels), dtype=np.float32)
        mask = np.zeros((n_samples, length), dtype=np.int64)
        lengths = np.zeros(n_samples, dtype=np.int64)

        unknown_bands: set = set()

        for i, sample in enumerate(samples):
            time, flux, flux_err, band = self._observations(sample, i)
            if time.size == 0:
                # Leaves an all-zero row masked off entirely, which prepare_inputs turns
                # into an all-NaN series. The model tolerates that; filter these objects
                # upstream in LSDB if you would rather not spend batch slots on them.
                continue

            keep = min(time.size, length)
            time, flux, flux_err = time[:keep], flux[:keep], flux_err[:keep]
            if band is not None:
                band = band[:keep]

            flux, flux_err = self._normalize_flux(flux, flux_err)

            # Log-compressed because season-length gaps otherwise dwarf intra-night cadence.
            gaps = np.zeros(keep, dtype=np.float64)
            if keep > 1:
                gaps[1:] = np.diff(time)
            gaps = np.log1p(np.clip(gaps, 0.0, None))

            series[i, :keep, FLUX_CHANNEL] = flux
            series[i, :keep, FLUX_ERR_CHANNEL] = flux_err
            series[i, :keep, DELTA_TIME_CHANNEL] = gaps

            if band is not None:
                band_indices, unknown = self._band_indices(band)
                unknown_bands |= unknown
                known = band_indices >= 0
                if known.any():
                    timesteps = np.arange(keep)[known]
                    series[i, timesteps, N_LEADING_CHANNELS + band_indices[known]] = 1.0

            mask[i, :keep] = 1
            lengths[i] = keep

        if unknown_bands:
            logger.warning(
                f"Ignoring observations in unrecognized band(s) {sorted(unknown_bands)}; they "
                f"get an all-zero band indicator. Configured bands: {self.bands}. Add them to "
                "config['data_set']['LightCurveLSDBStreamDataset']['bands'] to encode them."
            )

        result = {"series": series, "series_mask": mask, "series_lengths": lengths}
        result.update(self._passthrough_fields(samples))
        return result

    def _observations(self, sample: dict, index: int):
        """Return this object's finite observations, sorted by time.

        The returned band array is ``None`` for a single-band catalog (``band_field = false``).
        """
        time = np.asarray(sample[self.time_field], dtype=np.float64).ravel()
        flux = np.asarray(sample[self.flux_field], dtype=np.float64).ravel()
        flux_err = np.asarray(sample[self.flux_err_field], dtype=np.float64).ravel()
        band = None if self.band_field is None else np.asarray(sample[self.band_field]).ravel()

        observed = {
            self.time_field: time,
            self.flux_field: flux,
            self.flux_err_field: flux_err,
        }
        if band is not None:
            observed[self.band_field] = band

        sizes = {array.size for array in observed.values()}
        if len(sizes) > 1:
            n_obs = min(sizes)
            sizes_by_field = {name: int(array.size) for name, array in observed.items()}
            logger.warning(
                f"Light curve columns disagree in length for batch element {index} "
                f"({sizes_by_field}); truncating all of them to {n_obs}."
            )
            time, flux, flux_err = time[:n_obs], flux[:n_obs], flux_err[:n_obs]
            if band is not None:
                band = band[:n_obs]

        # Drop non-finite observations rather than carrying NaN into `series`: the mask is
        # the single source of truth for what is real, and a NaN here would trip
        # handle_nans on every batch.
        finite = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err)
        if not finite.all():
            logger.debug(f"Dropping {int((~finite).sum())} non-finite observations from element {index}.")
            time, flux, flux_err = time[finite], flux[finite], flux_err[finite]
            if band is not None:
                band = band[finite]

        order = np.argsort(time, kind="stable")
        return (
            time[order],
            flux[order],
            flux_err[order],
            None if band is None else band[order],
        )

    def _normalize_flux(self, flux: np.ndarray, flux_err: np.ndarray):
        """Center and rescale one object's flux, applying the same scale to its error.

        Per-object rather than per-batch: raw fluxes span orders of magnitude across bands
        and objects, and TS2Vec applies no normalization of its own.
        """
        if self.normalize is False:
            return flux, flux_err

        if self.normalize == "median_mad":
            center = float(np.median(flux))
            scale = MAD_TO_SIGMA * float(np.median(np.abs(flux - center)))
            if not scale > 0.0:
                # A MAD of zero means over half the points are identical; fall back to std.
                scale = float(np.std(flux))
        else:  # "zscore", validated in __init__
            center = float(np.mean(flux))
            scale = float(np.std(flux))

        if not scale > 0.0:
            # A constant light curve: centering still means something, rescaling does not.
            scale = 1.0

        # The same scale for both, so the error stays interpretable next to the flux.
        return (flux - center) / scale, flux_err / scale

    def _band_indices(self, band: np.ndarray):
        """Map band values to one-hot column offsets, returning ``-1`` for unrecognized ones.

        Accepts integer band codes directly and matches everything else against the
        configured band names. Mapping goes through ``np.unique`` so the dict lookups cost
        one per distinct band rather than one per observation.
        """
        if np.issubdtype(band.dtype, np.integer):
            codes = band.astype(np.int64)
            valid = (codes >= 0) & (codes < len(self.bands))
            unknown = {int(code) for code in np.unique(codes[~valid])} if not valid.all() else set()
            return np.where(valid, codes, -1), unknown

        keys = np.char.decode(band) if band.dtype.kind == "S" else band.astype(str)
        keys = np.char.strip(keys)

        uniques, inverse = np.unique(keys, return_inverse=True)
        mapped = np.array([self._band_index.get(str(u), -1) for u in uniques], dtype=np.int64)
        unknown = {str(u) for u, m in zip(uniques, mapped) if m < 0}
        return mapped[inverse], unknown

    def _passthrough_fields(self, samples: list[dict]) -> dict:
        """Collate every requested field that is not part of the light curve.

        Without this, defining a dataset-level ``collate`` would silently drop a requested
        ``ra``/``dec``, since it takes precedence over all field-level collation.
        """
        result = {}
        for field in samples[0]:
            if field in self.light_curve_fields:
                continue

            values = [np.asarray(sample[field]) for sample in samples]
            if len({value.shape for value in values}) != 1:
                logger.warning(
                    f"Skipping field '{field}': its values have differing shapes, so it "
                    "cannot be stacked. Add a collate function for it if you need it."
                )
                continue

            stacked = np.stack(values, axis=0)
            if stacked.dtype.kind not in "biufc":
                logger.warning(
                    f"Skipping non-numeric field '{field}' (dtype {stacked.dtype}); it cannot "
                    "be converted to a tensor."
                )
                continue
            result[field] = stacked

        return result

    def _check_fields(self, sample: dict) -> None:
        """Fail with an actionable message when a configured column is not in the data."""
        missing = [field for field in self.light_curve_fields if field not in sample]
        if missing:
            raise RuntimeError(
                f"LightCurveLSDBStreamDataset could not find the field(s) {missing} in the "
                f"data. Available fields: {sorted(sample)}. Nested columns are exposed as "
                "'get_<column>_<subcolumn>', so set time_field / flux_field / flux_err_field "
                "/ band_field in [data_set.LightCurveLSDBStreamDataset] to those flattened "
                "names, and list them in the data request's 'fields'."
            )
