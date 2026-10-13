"""Tests for HyraxTs2Vec's light-curve input encoding and its model interface.

Most of these drive :func:`build_event_sequence` directly with a hand-built collated batch,
which needs no catalog, no dataset and no model. The collated batch looks exactly like what
``LSDBStreamDataset``'s auto-registered nested collator produces: one ``<field>`` array
padded to the batch's longest row, plus a boolean ``<field>_mask``.
"""

import numpy as np
import pytest

import hyrax
from hyrax.context import clear_context, run_context
from hyrax.models.hyrax_ts2vec import (
    DELTA_TIME_CHANNEL,
    FLUX_CHANNEL,
    FLUX_ERR_CHANNEL,
    MAD_TO_SIGMA,
    N_LEADING_CHANNELS,
    EncodingSettings,
    HyraxTs2Vec,
    build_event_sequence,
)

TIME_FIELD = "lc_time"
FLUX_FIELD = "lc_flux"
FLUX_ERR_FIELD = "lc_flux_err"
BAND_FIELD = "lc_band"
BANDS = ["u", "g", "r", "i", "z", "y"]


def make_config(
    band_field=BAND_FIELD,
    bands=None,
    max_sequence_length=8,
    normalize="median_mad",
    **model_overrides,
):
    """A config carrying only the keys the encoding reads, plus any model overrides."""
    return {
        "model": {
            "HyraxTs2Vec": {
                "time_field": TIME_FIELD,
                "flux_field": FLUX_FIELD,
                "flux_err_field": FLUX_ERR_FIELD,
                "band_field": band_field,
                "bands": BANDS if bands is None else bands,
                "max_sequence_length": max_sequence_length,
                "normalize": normalize,
                **model_overrides,
            }
        }
    }


def collate(samples, with_band=True):
    """Pad ragged per-object arrays the way the LSDBStreamDataset collator does."""
    fields = [TIME_FIELD, FLUX_FIELD, FLUX_ERR_FIELD] + ([BAND_FIELD] if with_band else [])
    width = max((len(sample[field]) for sample in samples for field in fields), default=0)

    data = {}
    for field in fields:
        arrays = [np.asarray(sample[field]) for sample in samples]
        dtype = np.result_type(*(array.dtype for array in arrays))
        padded = np.zeros((len(samples), width), dtype=dtype)
        mask = np.zeros((len(samples), width), dtype=bool)
        for i, array in enumerate(arrays):
            padded[i, : len(array)] = array
            mask[i, : len(array)] = True
        data[field] = padded
        data[f"{field}_mask"] = mask

    return {"object_id": np.array([f"id{i}" for i in range(len(samples))]), "data": data}


def light_curve(times, fluxes=None, errors=None, bands=None):
    """One object's observations, as ragged 1-D arrays."""
    times = np.asarray(times, dtype=np.float32)
    n = times.size
    sample = {
        TIME_FIELD: times,
        FLUX_FIELD: np.arange(n, dtype=np.float32) if fluxes is None else np.asarray(fluxes, np.float32),
        FLUX_ERR_FIELD: np.ones(n, np.float32) if errors is None else np.asarray(errors, np.float32),
    }
    if bands is not None:
        # dtype=str rather than a fixed width, so a long band name is not silently clipped.
        sample[BAND_FIELD] = np.asarray(bands, dtype=str)
    return sample


#
# Channel layout and shape
#


def test_channel_count_is_three_plus_bands():
    """The one-hot block widens the input; nothing else changes."""
    samples = [light_curve([1.0, 2.0], bands=["g", "r"])]
    series = build_event_sequence(collate(samples), make_config())

    assert series.shape == (1, 8, N_LEADING_CHANNELS + len(BANDS))
    assert series.dtype == np.float32


def test_single_band_survey_drops_the_one_hot_block():
    """band_field = false leaves a series of just flux, flux error, and time gap."""
    samples = [light_curve([1.0, 2.0, 3.0])]
    series = build_event_sequence(collate(samples, with_band=False), make_config(band_field=False))

    assert series.shape == (1, 8, N_LEADING_CHANNELS)


def test_padding_is_nan_not_zero():
    """The encoder reads NaN as 'nothing here'; a zero would be a real measurement."""
    samples = [light_curve([1.0, 2.0], bands=["g", "r"])]
    series = build_event_sequence(collate(samples), make_config(normalize=False))

    assert np.isfinite(series[0, :2]).all()
    assert np.isnan(series[0, 2:]).all()


def test_short_curves_are_padded_to_max_sequence_length():
    """Every batch has the same time axis regardless of the longest curve in it."""
    samples = [light_curve([1.0, 2.0, 3.0], bands=["g"] * 3)]
    series = build_event_sequence(collate(samples), make_config(max_sequence_length=32))

    assert series.shape[1] == 32
    assert np.isnan(series[0, 3:]).all()


def test_long_curves_are_truncated_to_the_earliest_observations():
    """Truncation keeps the start of the curve, after sorting - not the start of the row."""
    times = [50.0, 10.0, 30.0, 20.0, 40.0]
    samples = [light_curve(times, fluxes=times, bands=["g"] * 5)]
    series = build_event_sequence(collate(samples), make_config(max_sequence_length=3, normalize=False))

    assert series.shape[1] == 3
    np.testing.assert_allclose(series[0, :, FLUX_CHANNEL], [10.0, 20.0, 30.0])


#
# Ordering
#


def test_observations_are_sorted_by_time():
    """Rows arrive in catalog order; the sequence is an ordered sequence of events."""
    times = [3.0, 1.0, 2.0]
    samples = [light_curve(times, fluxes=[30.0, 10.0, 20.0], bands=["r", "g", "i"])]
    series = build_event_sequence(collate(samples), make_config(normalize=False))

    np.testing.assert_allclose(series[0, :3, FLUX_CHANNEL], [10.0, 20.0, 30.0])
    # The band indicator travels with its observation.
    for step, band in enumerate(["g", "i", "r"]):
        assert series[0, step, N_LEADING_CHANNELS + BANDS.index(band)] == 1.0


def test_padding_sorts_past_real_observations():
    """A short curve in a wide batch keeps its observations at the front of the sequence."""
    samples = [
        light_curve([1.0, 2.0, 3.0, 4.0], bands=["g"] * 4),
        light_curve([9.0], fluxes=[42.0], bands=["r"]),
    ]
    series = build_event_sequence(collate(samples), make_config(normalize=False))

    np.testing.assert_allclose(series[1, 0, FLUX_CHANNEL], 42.0)
    assert np.isnan(series[1, 1:]).all()


#
# The time gap channel
#


def test_delta_time_channel_is_log1p_of_the_gap():
    """Log-compressed, because season-length gaps otherwise dwarf intra-night cadence."""
    times = [0.0, 1.0, 11.0]
    samples = [light_curve(times, bands=["g"] * 3)]
    series = build_event_sequence(collate(samples), make_config())

    gaps = series[0, :3, DELTA_TIME_CHANNEL]
    np.testing.assert_allclose(gaps, np.log1p([0.0, 1.0, 10.0]), rtol=1e-6)


def test_first_observation_has_no_gap():
    """There is no previous observation to measure from."""
    samples = [light_curve([100.0, 101.0], bands=["g", "g"])]
    series = build_event_sequence(collate(samples), make_config())

    assert series[0, 0, DELTA_TIME_CHANNEL] == 0.0


#
# Per-object flux normalization
#


def test_median_mad_normalization_centers_and_rescales():
    """Robust to an outburst: the median and MAD ignore the one extreme point."""
    fluxes = [10.0, 12.0, 14.0, 16.0, 500.0]
    samples = [light_curve(np.arange(5.0), fluxes=fluxes, errors=np.full(5, 2.0), bands=["g"] * 5)]
    series = build_event_sequence(collate(samples), make_config(normalize="median_mad"))

    center = np.median(fluxes)
    scale = MAD_TO_SIGMA * np.median(np.abs(np.asarray(fluxes) - center))
    np.testing.assert_allclose(series[0, :5, FLUX_CHANNEL], (np.asarray(fluxes) - center) / scale, rtol=1e-5)
    # The error is rescaled but not recentered, so it stays interpretable next to the flux.
    np.testing.assert_allclose(series[0, :5, FLUX_ERR_CHANNEL], 2.0 / scale, rtol=1e-5)


def test_zscore_normalization_uses_mean_and_std():
    """The plain alternative to median/MAD, for curves without outliers."""
    fluxes = [1.0, 2.0, 3.0, 4.0]
    samples = [light_curve(np.arange(4.0), fluxes=fluxes, bands=["g"] * 4)]
    series = build_event_sequence(collate(samples), make_config(normalize="zscore"))

    expected = (np.asarray(fluxes) - np.mean(fluxes)) / np.std(fluxes)
    np.testing.assert_allclose(series[0, :4, FLUX_CHANNEL], expected, rtol=1e-5)


def test_normalization_can_be_disabled():
    """normalize = false hands the raw fluxes straight through."""
    fluxes = [100.0, 200.0, 300.0]
    samples = [light_curve(np.arange(3.0), fluxes=fluxes, bands=["g"] * 3)]
    series = build_event_sequence(collate(samples), make_config(normalize=False))

    np.testing.assert_allclose(series[0, :3, FLUX_CHANNEL], fluxes)


def test_constant_light_curve_is_centered_but_not_rescaled():
    """A MAD of zero falls back to std, and a std of zero to a scale of one."""
    samples = [light_curve(np.arange(4.0), fluxes=np.full(4, 7.5), bands=["g"] * 4)]
    series = build_event_sequence(collate(samples), make_config(normalize="median_mad"))

    np.testing.assert_allclose(series[0, :4, FLUX_CHANNEL], 0.0, atol=1e-6)


def test_normalization_is_per_object_not_per_batch():
    """Two objects on wildly different flux scales normalize to the same range."""
    samples = [
        light_curve(np.arange(4.0), fluxes=[1.0, 2.0, 3.0, 4.0], bands=["g"] * 4),
        light_curve(np.arange(4.0), fluxes=[1e6, 2e6, 3e6, 4e6], bands=["g"] * 4),
    ]
    series = build_event_sequence(collate(samples), make_config(normalize="zscore"))

    np.testing.assert_allclose(series[0, :4, FLUX_CHANNEL], series[1, :4, FLUX_CHANNEL], rtol=1e-4)


def test_normalization_ignores_padding():
    """A short curve's statistics come from its own observations, not from the zero fill."""
    samples = [
        light_curve(np.arange(2.0), fluxes=[10.0, 20.0], bands=["g"] * 2),
        light_curve(np.arange(6.0), fluxes=np.arange(6.0) * 100, bands=["g"] * 6),
    ]
    series = build_event_sequence(collate(samples), make_config(normalize="zscore"))

    expected = (np.array([10.0, 20.0]) - 15.0) / np.std([10.0, 20.0])
    np.testing.assert_allclose(series[0, :2, FLUX_CHANNEL], expected, rtol=1e-5)


#
# Bands
#


def test_band_one_hot_is_placed_after_the_leading_channels():
    """The three leading channels are fixed; bands occupy everything past them."""
    samples = [light_curve([1.0, 2.0, 3.0], bands=["u", "r", "y"])]
    series = build_event_sequence(collate(samples), make_config())

    one_hot = series[0, :3, N_LEADING_CHANNELS:]
    assert one_hot.sum() == 3.0
    for step, band in enumerate(["u", "r", "y"]):
        assert one_hot[step, BANDS.index(band)] == 1.0


def test_integer_band_codes_index_the_configured_bands():
    """A catalog that stores bands as integer codes needs no upstream translation."""
    sample = light_curve([1.0, 2.0])
    sample[BAND_FIELD] = np.array([0, 3], dtype=np.int64)
    series = build_event_sequence(collate([sample]), make_config())

    assert series[0, 0, N_LEADING_CHANNELS + 0] == 1.0
    assert series[0, 1, N_LEADING_CHANNELS + 3] == 1.0


def test_unrecognized_band_gets_an_all_zero_indicator_and_a_warning(caplog):
    """The observation is kept - only its band is unknown."""
    samples = [light_curve([1.0, 2.0], bands=["g", "NOTABAND"])]
    series = build_event_sequence(collate(samples), make_config())

    assert series[0, 1, N_LEADING_CHANNELS:].sum() == 0.0
    assert np.isfinite(series[0, 1, FLUX_CHANNEL])
    assert "NOTABAND" in caplog.text


def test_padding_is_not_reported_as_an_unrecognized_band(caplog):
    """A string column pads with the empty string, which is not an observation."""
    samples = [
        light_curve([1.0, 2.0, 3.0], bands=["g", "r", "i"]),
        light_curve([1.0], bands=["g"]),
    ]
    build_event_sequence(collate(samples), make_config())

    assert "unrecognized band" not in caplog.text


#
# Validity and degenerate rows
#


def test_non_finite_observations_are_dropped():
    """NaN in the raw flux marks a missing measurement, not a value to carry through."""
    samples = [light_curve([1.0, 2.0, 3.0], fluxes=[10.0, np.nan, 30.0], bands=["g", "r", "i"])]
    series = build_event_sequence(collate(samples), make_config(normalize=False))

    np.testing.assert_allclose(series[0, :2, FLUX_CHANNEL], [10.0, 30.0])
    assert np.isnan(series[0, 2:]).all()


def test_object_with_no_observations_is_all_nan():
    """Left for the model to drop, rather than silently changing the batch size here."""
    samples = [light_curve([1.0, 2.0], bands=["g", "r"]), light_curve([], bands=[])]
    series = build_event_sequence(collate(samples), make_config())

    assert np.isnan(series[1]).all()
    assert np.isfinite(series[0, :2]).all()


def test_single_observation_curve():
    """No gap to compute and no spread to normalize by."""
    samples = [light_curve([5.0], fluxes=[42.0], bands=["g"])]
    series = build_event_sequence(collate(samples), make_config())

    assert np.isfinite(series[0, 0]).all()
    assert series[0, 0, DELTA_TIME_CHANNEL] == 0.0
    assert np.isnan(series[0, 1:]).all()


def test_missing_mask_treats_every_entry_as_an_observation():
    """A fixed-length nested column is stacked directly and has no mask."""
    batch = collate([light_curve([1.0, 2.0], bands=["g", "r"])])
    for field in (TIME_FIELD, FLUX_FIELD, FLUX_ERR_FIELD, BAND_FIELD):
        del batch["data"][f"{field}_mask"]

    series = build_event_sequence(batch, make_config(normalize=False))
    assert np.isfinite(series[0, :2]).all()


#
# Errors
#


def test_missing_field_names_the_field_and_what_is_available():
    """A misconfigured column name is the easiest mistake to make here."""
    batch = collate([light_curve([1.0, 2.0], bands=["g", "r"])])
    del batch["data"][FLUX_FIELD]

    with pytest.raises(RuntimeError, match=FLUX_FIELD):
        build_event_sequence(batch, make_config())


def test_missing_data_group_explains_the_naming_requirement():
    """The data request group must be named 'data' for prepare_inputs to find it."""
    batch = collate([light_curve([1.0, 2.0], bands=["g", "r"])])
    batch["lightcurves"] = batch.pop("data")

    with pytest.raises(RuntimeError, match="'data' key"):
        build_event_sequence(batch, make_config())


def test_mismatched_column_shapes_are_rejected():
    """Columns from different nested columns cannot describe one object's row."""
    batch = collate([light_curve([1.0, 2.0, 3.0], bands=["g", "r", "i"])])
    batch["data"][FLUX_FIELD] = batch["data"][FLUX_FIELD][:, :2]

    with pytest.raises(RuntimeError, match="common shape"):
        build_event_sequence(batch, make_config())


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"bands": []}, "must list at least one band"),
        ({"max_sequence_length": False}, "positive integer"),
        ({"max_sequence_length": 0}, "positive integer"),
        ({"normalize": "robust"}, "must be one of"),
    ],
)
def test_encoding_settings_reject_bad_values(overrides, match):
    """Bad settings fail during model pre-flight rather than midway through a run."""
    with pytest.raises(ValueError, match=match):
        EncodingSettings(make_config(**overrides))


#
# The model interface
#


@pytest.fixture
def hyrax_config():
    """A real Hyrax config, so the defaults under test are the shipped ones."""
    h = hyrax.Hyrax()
    model_config = h.config["model"]["HyraxTs2Vec"]
    model_config["time_field"] = TIME_FIELD
    model_config["flux_field"] = FLUX_FIELD
    model_config["flux_err_field"] = FLUX_ERR_FIELD
    model_config["band_field"] = BAND_FIELD
    model_config["max_sequence_length"] = 16
    # A shallow encoder keeps these tests fast; depth does not affect the input contract.
    model_config["depth"] = 2
    model_config["hidden_dims"] = 8
    model_config["output_dims"] = 4
    return h.config


@pytest.fixture
def _context(hyrax_config):
    """prepare_inputs reads the config from the run context, as it does inside a verb."""
    with run_context("test", config=hyrax_config):
        yield
    clear_context()


def test_default_config_ships_every_encoding_key():
    """EncodingSettings must be satisfiable from the packaged defaults alone."""
    settings = EncodingSettings(hyrax.Hyrax().config)

    assert settings.n_channels == N_LEADING_CHANNELS + len(settings.bands)
    assert settings.max_sequence_length > 0


def test_prepare_inputs_reads_the_config_from_the_run_context(hyrax_config, _context):
    """The whole reason the verb puts its config in the context."""
    batch = collate([light_curve([1.0, 2.0], bands=["g", "r"])])
    series = HyraxTs2Vec.prepare_inputs(batch)

    assert series.shape == (1, 16, N_LEADING_CHANNELS + len(hyrax_config["model"]["HyraxTs2Vec"]["bands"]))


def test_prepare_inputs_outside_a_verb_run_explains_itself():
    """The context is empty when a model is driven by hand, and says so."""
    clear_context()
    batch = collate([light_curve([1.0, 2.0], bands=["g", "r"])])

    with pytest.raises(KeyError, match="run context"):
        HyraxTs2Vec.prepare_inputs(batch)


def test_model_sizes_its_input_layer_from_the_prepared_sample(hyrax_config):
    """setup_model runs prepare_inputs before construction, so data_sample is the series."""
    samples = [light_curve(np.arange(8.0), bands=["g"] * 8)]
    data_sample = build_event_sequence(collate(samples), hyrax_config)
    model = HyraxTs2Vec(hyrax_config, data_sample=data_sample)

    assert model.input_dims == N_LEADING_CHANNELS + len(hyrax_config["model"]["HyraxTs2Vec"]["bands"])


def test_train_batch_runs_on_an_encoded_batch(hyrax_config):
    """End to end from collated columns to a loss, with no dataset subclass involved."""
    import torch

    rng = np.random.default_rng(0)
    samples = [
        light_curve(
            np.sort(rng.uniform(0.0, 100.0, 16)),
            fluxes=rng.normal(500.0, 50.0, 16),
            bands=rng.choice(BANDS, 16),
        )
        for _ in range(4)
    ]
    series = build_event_sequence(collate(samples), hyrax_config)
    model = HyraxTs2Vec(hyrax_config, data_sample=series)

    metrics = model.train_batch(torch.from_numpy(series))

    assert np.isfinite(metrics["loss"])


def test_infer_batch_keeps_every_row(hyrax_config):
    """Inference output must stay aligned with object_id, empty light curves included."""
    import torch

    samples = [light_curve(np.arange(8.0), bands=["g"] * 8), light_curve([], bands=[])]
    series = build_event_sequence(collate(samples), hyrax_config)
    model = HyraxTs2Vec(hyrax_config, data_sample=series)
    model.eval()

    with torch.no_grad():
        output = model.infer_batch(torch.from_numpy(series))

    assert output.shape[0] == 2
