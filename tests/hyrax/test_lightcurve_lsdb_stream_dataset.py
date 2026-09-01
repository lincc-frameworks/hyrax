"""Tests for LightCurveLSDBStreamDataset's collation.

These exercise `collate` directly on synthetic sample dicts. The stream machinery the parent
class provides (opening a HATS catalog, chunking partitions) needs lsdb and a real catalog
and is covered by tests/hyrax/test_lsdb_stream_dataset.py; the collation is pure array work,
so it is tested without constructing the parent at all.
"""

import logging

import numpy as np
import pytest

from hyrax.datasets.lightcurve_lsdb_stream_dataset import (
    DELTA_TIME_CHANNEL,
    FLUX_CHANNEL,
    FLUX_ERR_CHANNEL,
    N_LEADING_CHANNELS,
    LightCurveLSDBStreamDataset,
)

TIME_FIELD = "lightcurve_time"
FLUX_FIELD = "lightcurve_flux"
FLUX_ERR_FIELD = "lightcurve_flux_err"
BAND_FIELD = "lightcurve_band"

BANDS = ["u", "g", "r", "i", "z", "y"]
MAX_LENGTH = 8


def make_config(max_sequence_length=MAX_LENGTH, normalize="median_mad", bands=None, band_field=BAND_FIELD):
    """The `[data_set.LightCurveLSDBStreamDataset]` block, standing in for a full config."""
    return {
        "data_set": {
            "LightCurveLSDBStreamDataset": {
                "time_field": TIME_FIELD,
                "flux_field": FLUX_FIELD,
                "flux_err_field": FLUX_ERR_FIELD,
                "band_field": band_field,
                "bands": BANDS if bands is None else bands,
                "max_sequence_length": max_sequence_length,
                "normalize": normalize,
            }
        }
    }


@pytest.fixture
def dataset():
    """A dataset whose collation is configured but whose stream was never opened.

    `LightCurveLSDBStreamDataset.__init__` calls up into the parent, which requires lsdb and
    a real catalog. `collate` and its helpers touch none of that, so the instance is built
    without `__init__` and only the collation attributes are set.
    """
    return make_dataset(make_config())


def make_dataset(config):
    """Build a collation-only instance, bypassing the parent's stream setup."""
    instance = LightCurveLSDBStreamDataset.__new__(LightCurveLSDBStreamDataset)
    LightCurveLSDBStreamDataset._init_collation(instance, config)
    return instance


def make_sample(times, fluxes, flux_errs=None, bands=None, **extra):
    """One object's worth of ragged light-curve arrays, as the stream hands them over."""
    times = np.asarray(times, dtype=np.float64)
    fluxes = np.asarray(fluxes, dtype=np.float64)
    if flux_errs is None:
        flux_errs = np.full(times.shape, 0.1)
    if bands is None:
        bands = np.array(["r"] * len(times))
    sample = {
        TIME_FIELD: times,
        FLUX_FIELD: fluxes,
        FLUX_ERR_FIELD: np.asarray(flux_errs, dtype=np.float64),
        BAND_FIELD: np.asarray(bands),
    }
    sample.update(extra)
    return sample


#
# Shape, mask, and lengths
#


def test_collate_shapes_and_keys(dataset):
    """collate() emits the series, its mask, and its lengths at one fixed batch shape."""
    samples = [
        make_sample([0.0, 1.0, 2.0], [10.0, 11.0, 12.0]),
        make_sample([0.0, 5.0], [20.0, 25.0]),
    ]

    result = dataset.collate(samples)

    assert set(result) == {"series", "series_mask", "series_lengths"}
    assert result["series"].shape == (2, MAX_LENGTH, N_LEADING_CHANNELS + len(BANDS))
    assert result["series"].dtype == np.float32
    assert result["series_mask"].shape == (2, MAX_LENGTH)
    assert result["series_mask"].dtype == np.int64
    np.testing.assert_array_equal(result["series_lengths"], [3, 2])


def test_collate_pads_with_zeros_and_masks_the_padding(dataset):
    """Padding must be zeros, not NaN: Hyrax's handle_nans runs over this array and would
    either warn on every batch or overwrite the sentinel outright."""
    result = dataset.collate([make_sample([0.0, 1.0], [10.0, 12.0])])

    assert not np.isnan(result["series"]).any()
    np.testing.assert_array_equal(result["series"][0, 2:], 0.0)
    np.testing.assert_array_equal(result["series_mask"][0], [1, 1, 0, 0, 0, 0, 0, 0])


def test_collate_truncates_and_clamps_lengths(dataset):
    """`series_lengths` reports the kept count, so it always agrees with the mask."""
    n_obs = MAX_LENGTH + 5
    result = dataset.collate([make_sample(np.arange(n_obs), np.arange(n_obs) * 2.0)])

    assert result["series_lengths"][0] == MAX_LENGTH
    assert result["series_mask"][0].sum() == MAX_LENGTH


def test_collate_handles_an_object_with_no_observations(dataset):
    """An empty curve leaves a fully masked row, which prepare_inputs turns into all-NaN."""
    result = dataset.collate([make_sample([], []), make_sample([0.0, 1.0], [1.0, 2.0])])

    assert result["series_lengths"][0] == 0
    assert result["series_mask"][0].sum() == 0
    np.testing.assert_array_equal(result["series"][0], 0.0)
    assert result["series_lengths"][1] == 2


#
# Ordering, gaps, and bands
#


def test_collate_sorts_observations_by_time(dataset):
    """Out-of-order rows are common in nested catalogs, and the model reads position as time."""
    unsorted = make_sample([5.0, 1.0, 3.0], [50.0, 10.0, 30.0], bands=["z", "u", "g"])

    result = dataset.collate([unsorted])

    flux = result["series"][0, :3, FLUX_CHANNEL]
    assert flux[0] < flux[1] < flux[2]
    # The band one-hot has to travel with its observation, not stay at the original index.
    one_hot = result["series"][0, :3, N_LEADING_CHANNELS:]
    assert one_hot[0].argmax() == BANDS.index("u")
    assert one_hot[1].argmax() == BANDS.index("g")
    assert one_hot[2].argmax() == BANDS.index("z")


def test_delta_time_channel_is_log1p_of_the_gap(dataset):
    """The first observation has no predecessor, so its gap is zero by definition."""
    result = make_dataset(make_config(normalize=False)).collate(
        [make_sample([0.0, 1.0, 11.0], [1.0, 2.0, 3.0])]
    )

    gaps = result["series"][0, :3, DELTA_TIME_CHANNEL]
    assert gaps[0] == 0.0
    assert gaps[1] == pytest.approx(np.log1p(1.0), rel=1e-6)
    assert gaps[2] == pytest.approx(np.log1p(10.0), rel=1e-6)


def test_band_one_hot_is_exclusive(dataset):
    """Exactly one band channel is set for each observation."""
    result = dataset.collate([make_sample([0.0, 1.0], [1.0, 2.0], bands=["u", "y"])])

    one_hot = result["series"][0, :2, N_LEADING_CHANNELS:]
    np.testing.assert_array_equal(one_hot.sum(axis=1), [1.0, 1.0])
    assert one_hot[0].argmax() == 0
    assert one_hot[1].argmax() == len(BANDS) - 1


def test_integer_band_codes_are_read_as_indices(dataset):
    """Some catalogs store the band as a code rather than a name."""
    result = dataset.collate([make_sample([0.0, 1.0], [1.0, 2.0], bands=np.array([0, 3]))])

    one_hot = result["series"][0, :2, N_LEADING_CHANNELS:]
    assert one_hot[0].argmax() == 0
    assert one_hot[1].argmax() == 3


def test_byte_string_bands_are_decoded(dataset):
    """A byte-string band column is decoded, not stringified into its repr."""
    result = dataset.collate([make_sample([0.0], [1.0], bands=np.array([b"g"], dtype="S1"))])

    one_hot = result["series"][0, 0, N_LEADING_CHANNELS:]
    assert one_hot.argmax() == BANDS.index("g")


def test_unknown_band_gets_a_zero_indicator_and_a_warning(dataset, caplog):
    """An unconfigured filter must not silently masquerade as a configured one."""
    with caplog.at_level(logging.WARNING):
        result = dataset.collate([make_sample([0.0, 1.0], [1.0, 2.0], bands=["r", "Halpha"])])

    one_hot = result["series"][0, :2, N_LEADING_CHANNELS:]
    assert one_hot[0].sum() == 1.0
    assert one_hot[1].sum() == 0.0
    assert "Halpha" in caplog.text
    # The observation itself is still real, just unlabelled by band.
    assert result["series_mask"][0].sum() == 2


#
# Non-finite observations
#


def test_non_finite_observations_are_dropped(dataset):
    """Dropped rather than carried through as NaN, so the mask stays the only source of
    truth about what is real."""
    result = dataset.collate(
        [
            make_sample(
                [0.0, 1.0, 2.0, 3.0],
                [10.0, np.nan, 12.0, 13.0],
                flux_errs=[0.1, 0.1, np.inf, 0.1],
            )
        ]
    )

    assert result["series_lengths"][0] == 2
    assert not np.isnan(result["series"]).any()
    assert np.isfinite(result["series"]).all()


def test_mismatched_column_lengths_are_truncated_with_a_warning(dataset, caplog):
    """Ragged sibling columns are truncated to their common length, and it is reported."""
    sample = {
        TIME_FIELD: np.array([0.0, 1.0, 2.0]),
        FLUX_FIELD: np.array([1.0, 2.0]),
        FLUX_ERR_FIELD: np.array([0.1, 0.1, 0.1]),
        BAND_FIELD: np.array(["r", "r", "r"]),
    }

    with caplog.at_level(logging.WARNING):
        result = dataset.collate([sample])

    assert result["series_lengths"][0] == 2
    assert "disagree in length" in caplog.text


#
# Normalization
#


def test_median_mad_normalization_centers_the_flux(dataset):
    """The default normalization maps the median to zero and shrinks the spread to order unity."""
    fluxes = [100.0, 110.0, 120.0, 130.0, 140.0]
    result = dataset.collate([make_sample(np.arange(5.0), fluxes)])

    flux = result["series"][0, :5, FLUX_CHANNEL]
    # The median maps to zero, and the spread is now order-unity rather than order-100.
    assert flux[2] == pytest.approx(0.0, abs=1e-6)
    assert np.abs(flux).max() < 10.0


def test_flux_error_is_scaled_but_not_centered(dataset):
    """Sharing the flux's scale keeps the error interpretable next to it; centering an error
    would not make sense."""
    fluxes = [100.0, 110.0, 120.0, 130.0, 140.0]
    flux_errs = [10.0] * 5
    result = dataset.collate([make_sample(np.arange(5.0), fluxes, flux_errs=flux_errs)])

    flux_err = result["series"][0, :5, FLUX_ERR_CHANNEL]
    scale = 1.4826 * np.median(np.abs(np.array(fluxes) - 120.0))
    np.testing.assert_allclose(flux_err, 10.0 / scale, rtol=1e-5)
    assert (flux_err > 0).all()


def test_constant_flux_does_not_divide_by_zero(dataset):
    """A flat light curve has zero MAD and zero standard deviation; it must still collate."""
    result = dataset.collate([make_sample(np.arange(4.0), [5.0, 5.0, 5.0, 5.0])])

    flux = result["series"][0, :4, FLUX_CHANNEL]
    assert np.isfinite(flux).all()
    np.testing.assert_allclose(flux, 0.0)


def test_mostly_constant_flux_falls_back_to_std(dataset):
    """More than half the points identical makes the MAD zero, but the curve is not flat."""
    result = dataset.collate([make_sample(np.arange(5.0), [5.0, 5.0, 5.0, 5.0, 50.0])])

    flux = result["series"][0, :5, FLUX_CHANNEL]
    assert np.isfinite(flux).all()
    # The outlying point must remain distinguishable from the constant ones.
    assert flux[4] != pytest.approx(flux[0])


def test_zscore_normalization(dataset):
    """zscore normalization yields zero mean and unit standard deviation."""
    result = make_dataset(make_config(normalize="zscore")).collate(
        [make_sample(np.arange(5.0), [1.0, 2.0, 3.0, 4.0, 5.0])]
    )

    flux = result["series"][0, :5, FLUX_CHANNEL]
    assert flux.mean() == pytest.approx(0.0, abs=1e-6)
    assert flux.std() == pytest.approx(1.0, rel=1e-5)


def test_normalization_can_be_disabled(dataset):
    """normalize = false leaves the raw flux values untouched."""
    fluxes = [100.0, 110.0, 120.0]
    result = make_dataset(make_config(normalize=False)).collate([make_sample([0.0, 1.0, 2.0], fluxes)])

    np.testing.assert_allclose(result["series"][0, :3, FLUX_CHANNEL], fluxes, rtol=1e-6)


#
# Pass-through fields and configuration errors
#


def test_extra_numeric_fields_are_passed_through(dataset):
    """A dataset-level collate takes precedence over field-level collation, so requesting
    ra/dec alongside the light curve must not silently drop them."""
    samples = [
        make_sample([0.0, 1.0], [1.0, 2.0], ra=np.float64(10.5), dec=np.float64(-20.25)),
        make_sample([0.0, 1.0], [3.0, 4.0], ra=np.float64(11.5), dec=np.float64(-21.25)),
    ]

    result = dataset.collate(samples)

    np.testing.assert_allclose(result["ra"], [10.5, 11.5])
    np.testing.assert_allclose(result["dec"], [-20.25, -21.25])


def test_non_numeric_extra_fields_are_skipped_with_a_warning(dataset, caplog):
    """A string column cannot become a tensor, so it is dropped rather than crashing later."""
    samples = [make_sample([0.0], [1.0], survey=np.str_("lsst"))]

    with caplog.at_level(logging.WARNING):
        result = dataset.collate(samples)

    assert "survey" not in result
    assert "non-numeric" in caplog.text


def test_missing_configured_field_raises_an_actionable_error(dataset):
    """The most likely first failure: column names differ between catalogs."""
    sample = make_sample([0.0], [1.0])
    del sample[FLUX_FIELD]

    with pytest.raises(RuntimeError, match=FLUX_FIELD):
        dataset.collate([sample])


#
# Single-band catalogs
#


def test_single_band_catalog_drops_the_one_hot_block():
    """`band_field = false` is how a single-band survey (TESS, Kepler) is streamed.

    Such a catalog has no band column at all, so requiring one would make it unusable. The
    band block collapses away and the series is just flux, flux error, and time gap.
    """
    dataset = make_dataset(make_config(band_field=False))
    sample = make_sample([0.0, 1.0, 2.0], [10.0, 11.0, 12.0])
    del sample[BAND_FIELD]

    result = dataset.collate([sample])

    assert dataset.n_channels == N_LEADING_CHANNELS
    assert result["series"].shape == (1, MAX_LENGTH, N_LEADING_CHANNELS)
    assert result["series_lengths"][0] == 3
    assert np.isfinite(result["series"]).all()


def test_single_band_catalog_ignores_the_bands_list():
    """With no band column there is nothing to one-hot, so `bands` must not add channels."""
    dataset = make_dataset(make_config(band_field=False, bands=BANDS))

    assert dataset.bands == []
    assert dataset.n_channels == N_LEADING_CHANNELS


def test_single_band_catalog_does_not_require_a_band_column():
    """The band field must drop out of the required-field check, not just the channel layout."""
    dataset = make_dataset(make_config(band_field=False))

    assert BAND_FIELD not in dataset.light_curve_fields


def test_empty_bands_list_is_rejected():
    """An empty band list would leave the series with no band channels at all."""
    with pytest.raises(ValueError, match="bands"):
        make_dataset(make_config(bands=[]))


def test_max_sequence_length_must_be_positive():
    """`false` is the TOML "not set" sentinel, and bool is an int subclass, so a bare
    int(False) here would silently mean a zero-length sequence."""
    with pytest.raises(ValueError, match="max_sequence_length"):
        make_dataset(make_config(max_sequence_length=False))


def test_unknown_normalize_mode_is_rejected():
    """An unsupported normalization mode fails at construction, not mid-stream."""
    with pytest.raises(ValueError, match="normalize"):
        make_dataset(make_config(normalize="minmax"))
