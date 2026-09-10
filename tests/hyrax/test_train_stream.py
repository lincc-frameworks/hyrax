"""Tests for the train_stream verb's streaming training session.

Reuses the FakeConsumer/FakeMessage stand-ins from ``test_kafka_stream_dataset.py``
(patched onto ``KafkaStreamDataset`` at the class level, since ``train_stream`` builds
the dataset internally) and the trivial ``HyraxLoopback`` model, so no real broker or
weights are needed. The manual-path tests construct ``TrainStreamSession`` directly with a
spy ``process_func`` to exercise the empty / small-batch skipping logic in isolation.
"""

import json
from pathlib import Path

import pytest
from test_kafka_stream_dataset import FakeConsumer, FakeMessage  # noqa: I001

import hyrax
from hyrax.datasets.kafka_stream_dataset import KafkaStreamDataset
from hyrax.verbs.train_stream import TrainStreamSession


class _TrainImageStream(KafkaStreamDataset):
    """KafkaStreamDataset subclass exposing the `get_<field>` accessors the provider needs.

    ``StreamingDataProvider`` reads each requested field through ``get_<field>(sample)`` on
    the wrapped dataset, so a stream must define one getter per field it offers.

    The name must stay unique across the test suite: every HyraxDataset subclass is
    auto-registered under its class name and the last registration silently wins, so a
    class of the same name in another test module would take over the ``dataset_class``
    lookup below -- and the consumer patched here would never be used.
    """

    def get_image(self, sample):
        """Return the image payload of one decoded sample."""
        return sample["image"]

    def get_object_id(self, sample):
        """Return the object id of one decoded sample."""
        return str(sample["object_id"])


def _msg(object_id, image):
    return FakeMessage(json.dumps({"object_id": object_id, "image": image}))


def _stream_hyrax(tmp_path, monkeypatch, *, num_messages=3, batch_size=2):
    """Configure a Hyrax instance to run train_stream over a Kafka-backed data request."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["data_loader"]["batch_size"] = batch_size

    ds_config = h.config["data_set"]["KafkaStreamDataset"]
    ds_config["topics"] = "test-topic"
    ds_config["batch_flush_timeout"] = 0.0  # flush partial batches on the first empty poll

    h.config["data_request"] = {
        "train_stream": {
            "data": {
                "dataset_class": "_TrainImageStream",
                "data_location": "./",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    }

    # One FakeConsumer per dataset instance; on exhaustion it stops that stream so the
    # iteration terminates. _make_consumer receives `self` (the stream) when patched.
    messages = [_msg(f"id{i}", [[float(i)]]) for i in range(num_messages)]
    monkeypatch.setattr(
        _TrainImageStream,
        "_make_consumer",
        lambda self: FakeConsumer(messages, on_exhausted=self.stop),
    )
    return h


class _SpyTensorboard:
    """Stand-in for the global tensorboard writer that records add_scalar calls."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        """Record one logged scalar."""
        self.scalars.append((tag, value, step))


def _manual_session(
    tmp_path,
    process_func,
    *,
    min_batch_size=False,
    save_weights_every=False,
    data_loader=None,
    provider=None,
):
    """Build a TrainStreamSession directly (manual path) with a spy model and process_func."""

    class _SpyModel:
        def __init__(self):
            self.saved = []

        def save(self, path):
            self.saved.append(path)

    config = {
        "train_stream": {
            "weights_filename": "weights.pth",
            "save_weights_every": save_weights_every,
            "min_batch_size": min_batch_size,
        }
    }
    session = TrainStreamSession(
        process_func=process_func,
        model=_SpyModel(),
        config=config,
        results_dir=Path(tmp_path),
        data_loader=data_loader,
        provider=provider,
    )
    return session


@pytest.fixture(autouse=True)
def _end_leaked_mlflow_runs():
    """Close any MLflow run a failing test left open.

    ``TrainStream.run`` opens a run that spans the whole session, so a test that fails
    before ``close()`` would otherwise leak an active run into every later test.
    """
    yield
    import mlflow

    while mlflow.active_run() is not None:
        mlflow.end_run()


def test_train_stream_iterates_streaming_dataset(tmp_path, monkeypatch):
    """A configured [data_request.train_stream] yields (batch, metrics) and saves weights."""
    h = _stream_hyrax(tmp_path, monkeypatch)

    seen_ids = []
    with h.train_stream() as session:
        for batch, metrics in session:
            seen_ids.extend(list(batch["object_id"]))
            # HyraxLoopback.train_batch is a no-op that returns a loss dict per batch.
            assert "loss" in metrics
        results_dir = session._results_dir

    # The peeked sample (used for model pre-flighting) is not lost.
    assert sorted(seen_ids) == ["id0", "id1", "id2"]
    # Final weights were persisted on close.
    assert (results_dir / "example_model.pth").exists()


def test_train_stream_close_returns_model_and_is_idempotent(tmp_path):
    """close() returns the model and repeated calls are safe (no double-save error)."""
    calls = []

    def process_func(engine, batch):
        calls.append(batch)
        return {"loss": 1.0}

    session = _manual_session(tmp_path, process_func)
    model = session._model
    assert session.close() is model
    # Idempotent: second close returns the same model and does not raise.
    assert session.close() is model
    # save_weights was invoked on close.
    assert model.saved


def test_process_after_close_raises(tmp_path):
    """Calling process()/train_batch() after close() raises RuntimeError."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.process({"object_id": ["a", "b"]})


def test_empty_batch_is_skipped(tmp_path):
    """An empty batch is skipped: no training step, returns None."""
    calls = []

    def process_func(engine, batch):
        calls.append(batch)
        return {"loss": 1.0}

    session = _manual_session(tmp_path, process_func)
    assert session.process({"object_id": []}) is None
    assert calls == []


def test_min_batch_size_skips_small_batches(tmp_path):
    """Batches smaller than min_batch_size are skipped; larger ones are trained."""
    calls = []

    def process_func(engine, batch):
        calls.append(batch)
        return {"loss": 1.0}

    session = _manual_session(tmp_path, process_func, min_batch_size=2)

    # One-sample batch is below the threshold -> skipped.
    assert session.process({"object_id": ["a"]}) is None
    assert calls == []

    # Two-sample batch meets the threshold -> trained.
    result = session.process({"object_id": ["a", "b"]})
    assert result == {"loss": 1.0}
    assert len(calls) == 1


def test_save_weights_every(tmp_path):
    """Weights are checkpointed every N processed batches."""
    session = _manual_session(
        tmp_path,
        lambda engine, batch: {"loss": 1.0},
        save_weights_every=2,
    )
    model = session._model

    session.process({"object_id": ["a", "b"]})  # batch 1, no save
    assert model.saved == []
    session.process({"object_id": ["c", "d"]})  # batch 2, save
    assert len(model.saved) == 1


def test_session_without_source_is_not_iterable(tmp_path):
    """A session built without a data_loader (manual path) cannot be iterated."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    with pytest.raises(RuntimeError, match="no data source"):
        list(session)


def test_missing_train_stream_data_request_is_rejected():
    """Configuring some other data group but not train_stream fails with an actionable error."""
    h = hyrax.Hyrax()
    h.config["data_request"] = {
        "infer": {
            "data": {
                "dataset_class": "HyraxCifarDataSet",
                "data_location": "./",
                "primary_id_field": "object_id",
                "fields": ["image"],
            }
        }
    }
    with pytest.raises(RuntimeError, match="train_stream"):
        h.train_stream()


def test_warm_start_weights_are_loaded_for_the_train_stream_section(tmp_path, monkeypatch):
    """A configured model_weights_file warm-starts the model, read from [train_stream]."""
    from hyrax.models import model_utils

    h = _stream_hyrax(tmp_path, monkeypatch)
    weights = tmp_path / "warm_start.pth"
    weights.write_text("")  # HyraxLoopback.load is a no-op; contents are irrelevant
    h.config["train_stream"]["model_weights_file"] = str(weights)

    calls = []
    monkeypatch.setattr(
        model_utils,
        "load_model_weights",
        lambda config, model, verb: calls.append((config[verb]["model_weights_file"], verb)),
    )

    session = h.train_stream()
    try:
        assert calls == [(str(weights), "train_stream")]
        # The streaming session trains, so the model must be left in training mode
        # (dropout / batchnorm active) rather than in eval mode.
        assert session._model.training
    finally:
        session.close()


def test_missing_warm_start_weights_file_raises(tmp_path, monkeypatch):
    """A model_weights_file that does not exist fails before the session is handed back."""
    h = _stream_hyrax(tmp_path, monkeypatch)
    h.config["train_stream"]["model_weights_file"] = str(tmp_path / "not_there.pth")

    with pytest.raises(RuntimeError, match="does not exist"):
        h.train_stream()


def test_mlflow_run_uses_configured_run_name_and_ends_on_close(tmp_path, monkeypatch):
    """The session-spanning MLflow run takes its name from config and is closed by close()."""
    import mlflow

    h = _stream_hyrax(tmp_path, monkeypatch)
    h.config["train_stream"]["run_name"] = "streaming-run"
    h.config["train_stream"]["experiment_name"] = "streaming-experiment"

    session = h.train_stream()
    try:
        run = mlflow.active_run()
        assert run is not None, "no MLflow run spans the session"
        assert run.info.run_name == "streaming-run"
    finally:
        session.close()

    # A stream has no fixed end, so the run is only ended by close(); if that is missed the
    # run leaks into whatever the notebook does next.
    assert mlflow.active_run() is None


def test_mlflow_run_name_defaults_to_the_results_dir(tmp_path, monkeypatch):
    """With run_name unset (false), the timestamped results directory names the run."""
    import mlflow

    h = _stream_hyrax(tmp_path, monkeypatch)
    h.config["train_stream"]["run_name"] = False

    session = h.train_stream()
    try:
        assert mlflow.active_run().info.run_name == session._results_dir.name
    finally:
        session.close()


def test_metrics_reach_mlflow_only_while_a_run_is_active(tmp_path, monkeypatch):
    """Per-batch metrics are logged under a training/ tag, but only with a run open."""
    import mlflow

    logged = []
    monkeypatch.setattr(mlflow, "log_metrics", lambda metrics, step: logged.append((metrics, step)))
    monkeypatch.setattr(mlflow, "active_run", lambda: None)

    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 0.25})
    session.process({"object_id": ["a", "b"]})
    assert logged == [], "metrics were logged with no active MLflow run"

    monkeypatch.setattr(mlflow, "active_run", lambda: object())
    session.process({"object_id": ["c", "d"]})
    assert logged == [({"training/loss": 0.25}, 2)]


def test_metrics_are_logged_to_tensorboard_with_the_batch_as_step(tmp_path):
    """Every metric in the result is logged per batch, stepped by processed-batch count."""
    session = _manual_session(
        tmp_path,
        lambda engine, batch: {"loss": 0.5, "kl_divergence": 0.25},
        min_batch_size=2,
    )
    spy = _SpyTensorboard()
    session._tb_logger = spy

    session.process({"object_id": ["a", "b"]})
    session.process({"object_id": ["c"]})  # skipped: must not consume a step
    session.process({"object_id": ["d", "e"]})

    assert spy.scalars == [
        ("training/training/loss", 0.5, 1),
        ("training/training/kl_divergence", 0.25, 1),
        ("training/training/loss", 0.5, 2),
        ("training/training/kl_divergence", 0.25, 2),
    ]


def test_non_dict_training_result_is_returned_without_logging(tmp_path):
    """A model whose train_batch returns a bare loss is passed through, not logged."""
    session = _manual_session(tmp_path, lambda engine, batch: 0.5)
    spy = _SpyTensorboard()
    session._tb_logger = spy

    assert session.process({"object_id": ["a", "b"]}) == 0.5
    assert spy.scalars == []


# NOTE: I'm unsure if this test is useful. All batches should be a dictionary and
# should have an "object_id" key that is inserted by Hyrax. So the branch of code
# this is testing should never happen in practice.
@pytest.mark.parametrize(
    "batch",
    [
        pytest.param({"data": [1, 2, 3]}, id="dict-without-object-id"),
        pytest.param(("image_tensor", "label_tensor"), id="non-dict-batch"),
    ],
)
def test_batches_of_unknown_size_are_always_trained(tmp_path, batch):
    """When the sample count cannot be read from the batch, min_batch_size cannot skip it."""
    calls = []

    def process_func(engine, batch):
        calls.append(batch)
        return {"loss": 1.0}

    session = _manual_session(tmp_path, process_func, min_batch_size=8)

    assert session.process(batch) == {"loss": 1.0}
    assert calls == [batch]


# NOTE: I'm unsure if this test is useful for that same reason as the previous one.
def test_skipped_batches_do_not_advance_the_save_counter(tmp_path):
    """save_weights_every counts trained batches, not batches offered to the session."""
    session = _manual_session(
        tmp_path,
        lambda engine, batch: {"loss": 1.0},
        min_batch_size=2,
        save_weights_every=2,
    )
    model = session._model

    session.process({"object_id": ["a", "b"]})  # trained batch 1
    session.process({"object_id": ["c"]})  # skipped
    assert model.saved == []
    session.process({"object_id": ["d", "e"]})  # trained batch 2 -> save
    assert len(model.saved) == 1


def test_iteration_yields_none_metrics_for_skipped_batches(tmp_path):
    """Iteration pairs every batch from the source with its metrics, or None if skipped."""
    batches = [{"object_id": ["a"]}, {"object_id": ["b", "c"]}]
    session = _manual_session(
        tmp_path,
        lambda engine, batch: {"loss": 1.0},
        min_batch_size=2,
        data_loader=batches,
    )

    assert list(session) == [(batches[0], None), (batches[1], {"loss": 1.0})]


def test_checkpoint_saves_under_a_loss_tagged_file_name(tmp_path):
    """A checkpoint is written beside the weights file, tagged with the loss it achieved."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    model = session._model

    session.checkpoint({"loss": 0.5})

    assert [path.name for path in model.saved] == ["weights_checkpoint_loss_0.5.pth"]
    assert model.saved[0].parent == Path(tmp_path)


def test_checkpoint_only_saves_when_the_loss_improves(tmp_path):
    """Checkpoints track the best loss so far; worse or equal losses are not written."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    model = session._model

    session.checkpoint({"loss": 0.5})  # first loss: best so far
    session.checkpoint({"loss": 0.9})  # worse -> skipped
    session.checkpoint({"loss": 0.5})  # equal to the best -> skipped
    session.checkpoint({"loss": 0.2})  # better -> saved

    assert [path.name for path in model.saved] == [
        "weights_checkpoint_loss_0.5.pth",
        "weights_checkpoint_loss_0.2.pth",
    ]


def test_checkpoint_treats_a_zero_best_loss_as_a_real_best(tmp_path):
    """A best loss of exactly 0.0 still gates later checkpoints.

    ``HyraxLoopback`` reports ``loss: 0.0``, and a perfectly-fit model reaches it, so the
    best-loss guard has to distinguish "no best yet" from "best is zero" rather than
    treating the falsy 0.0 as unset.
    """
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    model = session._model

    session.checkpoint({"loss": 0.0})
    session.checkpoint({"loss": 0.4})  # worse than a zero best -> skipped

    assert [path.name for path in model.saved] == ["weights_checkpoint_loss_0.0.pth"]
    assert session._best_loss == 0.0


@pytest.mark.parametrize(
    "model_metrics",
    [
        pytest.param(None, id="no-metrics"),
        pytest.param({"loss": None}, id="loss-is-none"),
        pytest.param({"accuracy": 0.9}, id="metrics-without-loss"),
    ],
)
def test_checkpoint_without_a_loss_saves_the_plain_weights_file(tmp_path, model_metrics):
    """With no loss to compare against, the checkpoint is unconditional and untagged."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    model = session._model

    session.checkpoint(model_metrics)

    assert [path.name for path in model.saved] == ["weights.pth"]


def test_close_stops_the_data_source_before_persisting_weights(tmp_path):
    """close() halts the stream first, so the final weights are not racing a live batch."""
    events = []

    class _SpyProvider:
        def stop(self):
            events.append("stop")

    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0}, provider=_SpyProvider())
    session._model.save = lambda path: events.append("save")

    session.close()

    assert events[0] == "stop"
    assert "save" in events[1:], "no weights were written on close"
    assert events.count("stop") == 1


def test_close_tolerates_a_provider_without_stop(tmp_path):
    """Only streaming providers have stop(); a map-style provider must not break close()."""

    class _MapStyleProvider:
        """Stands in for a plain DataProvider, which has no stop()."""

    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0}, provider=_MapStyleProvider())

    assert session.close() is session._model
    assert session._model.saved


def test_weights_are_persisted_when_the_with_body_raises(tmp_path):
    """An exception inside the `with` block still closes the session and saves the model."""
    session = _manual_session(tmp_path, lambda engine, batch: {"loss": 1.0})
    model = session._model

    with pytest.raises(ValueError, match="pipeline blew up"):
        with session:
            session.process({"object_id": ["a", "b"]})
            raise ValueError("pipeline blew up")

    assert model.saved, "training progress was lost when the body raised"
    assert session._closed
