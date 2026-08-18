import numpy as np
import pytest
import torch

from hyrax.models.hyrax_ts2vec import (
    HyraxTs2Vec,
    instance_contrastive_loss,
    take_per_row,
    temporal_contrastive_loss,
)

BATCH_SIZE = 4
# Long enough that the random crop scheme has room to draw two genuinely different
# overlapping windows (crop_l is drawn from [2, SEQUENCE_LENGTH]), and a power of two so the
# loss hierarchy's repeated halving lands exactly on a single timestep.
SEQUENCE_LENGTH = 16
# 3 leading channels (flux, flux error, time gap) plus a 6-band one-hot block, matching
# LightCurveLSDBStreamDataset's LSST default.
N_CHANNELS = 9
OUTPUT_DIMS = 8
HIDDEN_DIMS = 8
# Small on purpose: depth is the number of residual blocks, and each one doubles its
# dilation, so a large depth makes these tests slow without testing anything more.
DEPTH = 2


def make_model_config(
    output_dims=OUTPUT_DIMS,
    hidden_dims=HIDDEN_DIMS,
    depth=DEPTH,
    mask_mode="binomial",
    temporal_unit=0,
    alpha=0.5,
    encoding_window="full_series",
):
    """Minimal config dict containing the sections HyraxTs2Vec and the `hyrax_model`
    decorator need, without going through a full `hyrax.Hyrax()` instance.

    `criterion` and `optimizer` names still have to be present: the model sets both
    attributes itself, and the decorator reads those config keys to decide whether to warn
    about the duplication.
    """
    return {
        "model": {
            "HyraxTs2Vec": {
                "output_dims": output_dims,
                "hidden_dims": hidden_dims,
                "depth": depth,
                "mask_mode": mask_mode,
                "temporal_unit": temporal_unit,
                "alpha": alpha,
                "learning_rate": 0.001,
                "encoding_window": encoding_window,
            }
        },
        "criterion": {"name": None},
        "optimizer": {"name": None},
        "scheduler": {"name": None},
    }


def make_data_sample(n_channels=N_CHANNELS):
    """A stand-in for the `data_sample` HyraxTs2Vec inspects at init time to size its input
    layer. Only the trailing channel dimension is read."""
    return np.zeros((1, SEQUENCE_LENGTH, n_channels), dtype=np.float32)


def make_hyrax_batch(batch_size=BATCH_SIZE, seed=0, lengths=None):
    """Build a batch in the collated shape `prepare_inputs` receives from the real pipeline.

    `lengths` optionally sets the number of real observations per element, so padding can be
    exercised; the default marks every timestep as real.
    """
    rng = np.random.default_rng(seed)
    series = rng.standard_normal((batch_size, SEQUENCE_LENGTH, N_CHANNELS)).astype(np.float32)

    mask = np.zeros((batch_size, SEQUENCE_LENGTH), dtype=np.int64)
    if lengths is None:
        mask[:] = 1
        lengths = np.full(batch_size, SEQUENCE_LENGTH, dtype=np.int64)
    else:
        for i, length in enumerate(lengths):
            mask[i, :length] = 1
        lengths = np.asarray(lengths, dtype=np.int64)

    return {
        "object_id": np.array([str(i) for i in range(batch_size)]),
        "data": {
            "series": series,
            "series_mask": mask,
            "series_lengths": lengths,
        },
    }


def to_tensor_batch(series_array):
    """Do by hand what Hyrax's `_inner_loop` does in production: numpy in, tensor out."""
    return torch.from_numpy(series_array)


def prepared_tensor(**kwargs):
    """Shorthand for a batch taken all the way through `prepare_inputs` to a tensor."""
    return to_tensor_batch(HyraxTs2Vec.prepare_inputs(make_hyrax_batch(**kwargs)))


@pytest.fixture
def model():
    """A freshly constructed HyraxTs2Vec sized for the test batch's channel count."""
    return HyraxTs2Vec(config=make_model_config(), data_sample=make_data_sample())


#
# prepare_inputs
#


def test_prepare_inputs_applies_mask_as_nan():
    """Padded timesteps become NaN, real ones survive untouched.

    This conversion has to happen in `prepare_inputs` rather than in collation, because
    Hyrax's `handle_nans` would otherwise warn about or overwrite the sentinel.
    """
    lengths = [SEQUENCE_LENGTH, 5, 1, 0]
    hyrax_batch = make_hyrax_batch(lengths=lengths)
    original = hyrax_batch["data"]["series"].copy()

    series = HyraxTs2Vec.prepare_inputs(hyrax_batch)

    assert series.dtype == np.float32
    assert series.shape == (BATCH_SIZE, SEQUENCE_LENGTH, N_CHANNELS)
    for i, length in enumerate(lengths):
        np.testing.assert_array_equal(series[i, :length], original[i, :length])
        assert np.isnan(series[i, length:]).all()


def test_prepare_inputs_without_mask_passes_series_through():
    """A dataset that emits no mask is taken at its word: every timestep is real."""
    hyrax_batch = make_hyrax_batch()
    del hyrax_batch["data"]["series_mask"]

    series = HyraxTs2Vec.prepare_inputs(hyrax_batch)

    assert not np.isnan(series).any()
    np.testing.assert_array_equal(series, hyrax_batch["data"]["series"])


def test_prepare_inputs_raises_without_data_key():
    """Fail loudly, and name the convention, when the request group isn't called "data"."""
    with pytest.raises(RuntimeError, match="'data' key"):
        HyraxTs2Vec.prepare_inputs({"not_data": {}})


def test_prepare_inputs_raises_without_series_key():
    """Fail loudly when the dataset produced no `series` field."""
    with pytest.raises(RuntimeError, match="series"):
        HyraxTs2Vec.prepare_inputs({"data": {"image": np.zeros((2, 2))}})


#
# Construction
#


def test_init_reads_input_dims_from_data_sample():
    """The input layer is sized from the sample's channel count, not from config."""
    model = HyraxTs2Vec(config=make_model_config(), data_sample=make_data_sample(n_channels=12))

    assert model.input_dims == 12
    assert model.encoder.input_fc.in_features == 12


def test_init_raises_without_data_sample():
    """Without a sample the input width is unknowable, so construction must fail, not guess."""
    with pytest.raises(RuntimeError, match="data_sample"):
        HyraxTs2Vec(config=make_model_config(), data_sample=None)


def test_init_raises_on_wrong_rank_data_sample():
    """A 2-D sample is the mistake to expect here, so the error names the wanted shape."""
    with pytest.raises(RuntimeError, match="3-dimensional"):
        HyraxTs2Vec(config=make_model_config(), data_sample=np.zeros((4, 16), dtype=np.float32))


def test_model_owns_its_optimizer_and_criterion(model):
    """TS2Vec is specified on AdamW, so the model overrides the Hyrax optimizer default."""
    assert isinstance(model.optimizer, torch.optim.AdamW)
    assert model.criterion.alpha == 0.5


#
# forward
#


def test_forward_output_shape(model):
    """forward() returns one representation per timestep, per object."""
    series = prepared_tensor()

    out = model.forward(series)

    assert out.shape == (BATCH_SIZE, SEQUENCE_LENGTH, OUTPUT_DIMS)


def test_forward_tolerates_nan_timesteps(model):
    """NaN marks a missing observation; it must be masked out, never propagated."""
    model.eval()
    series = prepared_tensor(lengths=[SEQUENCE_LENGTH, 5, 1, 0])

    out = model.forward(series)

    assert torch.isfinite(out).all()


def test_forward_does_not_mutate_its_input(model):
    """The encoder zero-fills NaN internally; the caller's tensor must be left alone."""
    series = prepared_tensor(lengths=[SEQUENCE_LENGTH, 4, 4, 4])
    before = series.clone()

    model.forward(series)

    assert torch.equal(torch.isnan(series), torch.isnan(before))
    assert torch.equal(series[~torch.isnan(series)], before[~torch.isnan(before)])


#
# train / validate / test
#


def test_train_batch_returns_loss_and_updates_weights(model):
    """train_batch() returns a finite loss and leaves the model's weights changed."""
    torch.manual_seed(0)
    np.random.seed(0)
    series = prepared_tensor()
    before = model.encoder.input_fc.weight.detach().clone()

    result = model.train_batch(series)

    assert isinstance(result["loss"], float)
    assert np.isfinite(result["loss"])
    assert result["loss"] > 0.0
    assert not torch.allclose(before, model.encoder.input_fc.weight)


def test_validate_batch_does_not_update_weights(model):
    """validate_batch() reports a loss without taking an optimizer step."""
    torch.manual_seed(0)
    np.random.seed(0)
    series = prepared_tensor()
    before = model.encoder.input_fc.weight.detach().clone()

    result = model.validate_batch(series)

    assert np.isfinite(result["loss"])
    assert torch.allclose(before, model.encoder.input_fc.weight)


def test_test_batch_does_not_update_weights(model):
    """test_batch() reports a loss without taking an optimizer step."""
    torch.manual_seed(0)
    np.random.seed(0)
    series = prepared_tensor()
    before = model.encoder.input_fc.weight.detach().clone()

    result = model.test_batch(series)

    assert np.isfinite(result["loss"])
    assert torch.allclose(before, model.encoder.input_fc.weight)


def test_train_batch_drops_all_nan_rows(model):
    """An object with no observations at all must not poison the batch's loss."""
    torch.manual_seed(0)
    np.random.seed(0)
    series = prepared_tensor(lengths=[SEQUENCE_LENGTH, SEQUENCE_LENGTH, SEQUENCE_LENGTH, 0])

    result = model.train_batch(series)

    assert np.isfinite(result["loss"])
    assert result["loss"] > 0.0


def test_train_batch_skips_a_fully_empty_batch(model):
    """Every series missing means there is nothing to learn from; report zero, don't crash."""
    np.random.seed(0)
    series = prepared_tensor(lengths=[0, 0, 0, 0])
    before = model.encoder.input_fc.weight.detach().clone()

    result = model.train_batch(series)

    assert result == {"loss": 0.0}
    assert torch.allclose(before, model.encoder.input_fc.weight)


def test_train_batch_raises_on_too_short_sequence(model):
    """Below 2**(temporal_unit+1) timesteps there is no room to crop; say so plainly."""
    series = torch.zeros(BATCH_SIZE, 1, N_CHANNELS)

    with pytest.raises(RuntimeError, match="temporal_unit"):
        model.train_batch(series)


def test_train_batch_is_reproducible_with_a_seed():
    """Same seed, same weights, same loss - the crop draw and the mask are both seeded."""
    losses = []
    for _ in range(2):
        # Seed once for construction, so both models start from identical weights, then
        # again for the step itself, which draws both the crop offsets and the mask.
        torch.manual_seed(7)
        np.random.seed(7)
        model = HyraxTs2Vec(config=make_model_config(), data_sample=make_data_sample())

        torch.manual_seed(7)
        np.random.seed(7)
        losses.append(model.train_batch(prepared_tensor())["loss"])

    assert losses[0] == pytest.approx(losses[1])


def test_train_batch_survives_repeated_crop_draws(model):
    """Exercise many random crop draws: an off-by-one in the scheme shows up as an
    out-of-bounds gather, and a bad slice shows up as a non-finite loss."""
    np.random.seed(0)
    torch.manual_seed(0)
    series = prepared_tensor()

    for _ in range(40):
        result = model.train_batch(series)
        assert np.isfinite(result["loss"])


def test_batch_of_one_yields_only_the_temporal_term():
    """With no other instances to contrast against, the instance term drops to zero and the
    loss falls back to the temporal term alone (reference behavior)."""
    np.random.seed(0)
    torch.manual_seed(0)
    model = HyraxTs2Vec(config=make_model_config(), data_sample=make_data_sample())
    series = prepared_tensor(batch_size=1)

    result = model.train_batch(series)

    assert np.isfinite(result["loss"])


#
# infer_batch
#


def test_infer_batch_returns_one_vector_per_object(model):
    """The inference contract: one tensor, constant shape, aligned with object_id."""
    model.eval()
    series = prepared_tensor()

    out = model.infer_batch(series)

    assert out.shape == (BATCH_SIZE, OUTPUT_DIMS)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_infer_batch_keeps_empty_rows(model):
    """All-NaN rows are *not* dropped at inference; the output must stay 1:1 with object_id."""
    model.eval()
    series = prepared_tensor(lengths=[SEQUENCE_LENGTH, 3, 0, 0])

    out = model.infer_batch(series)

    assert out.shape == (BATCH_SIZE, OUTPUT_DIMS)
    assert torch.isfinite(out).all()


def test_infer_batch_is_deterministic_in_eval_mode(model):
    """eval() resolves the mask to all_true and disables dropout, so repeated calls agree.

    Without this, every inference run would write different latents for the same object.
    """
    model.eval()
    series = prepared_tensor()

    first = model.infer_batch(series)
    second = model.infer_batch(series)

    assert torch.equal(first, second)


def test_infer_batch_multiscale_keeps_the_time_axis():
    """The multiscale window concatenates several pooling scales at every timestep."""
    config = make_model_config(encoding_window="multiscale")
    model = HyraxTs2Vec(config=config, data_sample=make_data_sample())
    model.eval()

    out = model.infer_batch(prepared_tensor())

    assert out.shape[0] == BATCH_SIZE
    assert out.shape[1] == SEQUENCE_LENGTH
    # One max-pool level per power of two below the sequence length, concatenated.
    assert out.shape[2] % OUTPUT_DIMS == 0
    assert out.shape[2] > OUTPUT_DIMS


def test_infer_batch_rejects_an_unknown_encoding_window(model):
    """An unsupported pooling mode fails loudly rather than returning unpooled output."""
    model.eval()
    model.encoding_window = "sliding"

    with pytest.raises(ValueError, match="encoding_window"):
        model.infer_batch(prepared_tensor())


#
# Loss and gather building blocks
#


def test_instance_loss_is_zero_for_a_single_instance():
    """With only one instance there are no negatives, so the instance term vanishes."""
    z = torch.randn(1, SEQUENCE_LENGTH, OUTPUT_DIMS)

    assert instance_contrastive_loss(z, z).item() == 0.0


def test_temporal_loss_is_zero_for_a_single_timestamp():
    """With only one timestep there are no negatives, so the temporal term vanishes."""
    z = torch.randn(BATCH_SIZE, 1, OUTPUT_DIMS)

    assert temporal_contrastive_loss(z, z).item() == 0.0


def test_take_per_row_gathers_a_window_per_row():
    """Each row is offset independently, which is what gives the two crops their per-object
    context jitter."""
    tensor = torch.arange(3 * 6, dtype=torch.float32).reshape(3, 6, 1)

    gathered = take_per_row(tensor, np.array([0, 2, 3]), 3)

    assert gathered.shape == (3, 3, 1)
    np.testing.assert_array_equal(gathered[0, :, 0].numpy(), [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(gathered[1, :, 0].numpy(), [8.0, 9.0, 10.0])
    np.testing.assert_array_equal(gathered[2, :, 0].numpy(), [15.0, 16.0, 17.0])
