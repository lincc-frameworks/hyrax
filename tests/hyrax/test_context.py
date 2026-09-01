import inspect
import json
from pathlib import Path

import pytest
import torch.nn as nn

import hyrax
from hyrax.config_utils import find_most_recent_results_dir
from hyrax.context import (
    RunContext,
    clear_context,
    get_context,
    run_context,
    update_context,
    use_context,
)
from hyrax.models.model_registry import hyrax_model


def test_context_is_a_dict():
    """The context must be an ordinary dict so it can be passed to anything that
    expects one - vector databases, the ONNX exporter, ** unpacking, pickling."""
    context = get_context()
    assert isinstance(context, dict)
    assert isinstance(context, RunContext)
    assert context == {}


def test_context_identity_is_stable_within_a_run():
    """Everything reading the context during one run must see the same object, so a
    key one piece of code adds is visible to the rest of the run."""
    with run_context("train", results_dir="/some/results/dir") as context:
        assert get_context() is context
        update_context(user_key="stashed by a model")
        assert get_context()["user_key"] == "stashed by a model"


def test_each_run_gets_its_own_context_object():
    """The whole point of the per-run lifecycle: an object captured during one run is
    not the object a later run uses, so holding it cannot follow the wrong verb."""
    with run_context("train", results_dir="/first/run") as first:
        pass

    with run_context("infer", results_dir="/second/run") as second:
        assert second is not first

    assert first["results_dir"] == Path("/first/run")
    assert first["verb"] == "train"


def test_run_context_populates_expected_keys():
    """run_context plus the verb's own update_context fill the documented key set."""
    with run_context("infer", results_dir="/some/results/dir") as context:
        assert context["results_dir"] == Path("/some/results/dir")
        assert context["verb"] == "infer"
        assert context["rank"] == 0
        assert context["world_size"] == 1


def test_run_context_coerces_results_dir_to_path():
    """save_to_database used to pass a str; model_exporters uses the `/` operator."""
    with run_context("vector-db", results_dir="/some/results/dir") as context:
        assert isinstance(context["results_dir"], Path)

    with run_context("vector-db", results_dir=Path("/some/results/dir")) as context:
        assert isinstance(context["results_dir"], Path)


def test_run_context_accepts_extra_keys():
    """The to_onnx verb adds ml_framework this way."""
    with run_context("onnx", results_dir="/some/results/dir", ml_framework="pytorch") as context:
        assert context["ml_framework"] == "pytorch"


def test_run_context_releases_on_exit():
    """A finished run leaves nothing behind for the next one to inherit."""
    with run_context("train", results_dir="/some/run"):
        assert get_context()["verb"] == "train"

    assert get_context() == {}


def test_run_context_releases_when_the_body_raises():
    """A failed run must not leave the context pointing at its half-built directory."""
    with pytest.raises(ValueError):
        with run_context("train", results_dir="/some/run"):
            raise ValueError("something went wrong mid-run")

    assert get_context() == {}


def test_update_context_preserves_existing_keys():
    """train.py's _training updates the context in place, so anything a model stashed
    during setup_model survives on the single-process path."""
    with run_context("train", results_dir="/some/run"):
        update_context(user_key="stashed by a model")

        update_context(results_dir=Path("/some/run"), verb="train", rank=1, world_size=4)

        assert get_context()["user_key"] == "stashed by a model"
        assert get_context()["rank"] == 1
        assert get_context()["world_size"] == 4


def test_update_context_coerces_results_dir_to_path():
    """Coercion applies wherever values enter the context."""
    update_context(results_dir="/some/run")
    assert get_context()["results_dir"] == Path("/some/run")


def test_use_context_reinstalls_a_captured_context():
    """How an object that outlives its run - InferStreamSession - gives the model code
    it drives the run it actually belongs to."""
    with run_context("infer_stream", results_dir="/some/run") as captured:
        pass

    assert get_context() == {}

    with use_context(captured):
        assert get_context() is captured
        assert get_context()["results_dir"] == Path("/some/run")

    assert get_context() == {}


def test_missing_key_raises_helpful_error():
    """Hyrax users are scientists, not software engineers. A bare KeyError from a
    dict lookup would not tell them why the context was empty."""
    with run_context("train", results_dir="/some/run"):
        with pytest.raises(KeyError) as excinfo:
            get_context()["not_a_real_key"]

    message = str(excinfo.value)
    assert "not_a_real_key" in message
    assert "results_dir" in message  # lists the available keys
    assert "verb" in message


def test_missing_key_error_when_context_is_empty():
    """The empty-context case is the one users are most likely to hit."""
    with pytest.raises(KeyError) as excinfo:
        get_context()["results_dir"]

    assert "outside of a verb run" in str(excinfo.value)


def test_clear_context():
    """clear_context releases a context populated outside a verb run."""
    update_context(results_dir="/some/run", verb="train")
    clear_context()
    assert get_context() == {}


def test_get_context_is_exported_from_package():
    """Users are told to write `from hyrax import get_context`."""
    assert hyrax.get_context is get_context


def test_model_built_outside_a_verb_gets_empty_context():
    """Backward compatibility: constructing a model directly (as many existing tests
    do) must keep working, and reading the context there must explain itself."""

    @hyrax_model
    class ContextlessModel(nn.Module):
        def __init__(self, config, data_sample=None):
            super().__init__()
            self.config = config
            self.unused_module = nn.Linear(1, 1)
            self.context = get_context()

        def forward(self, x):
            return x

        def train_batch(self, batch):
            return {"loss": 0.0}

        def infer_batch(self, batch):
            return self.forward(batch)

    config = {
        "model": {},
        "criterion": {"name": "torch.nn.CrossEntropyLoss"},
        "optimizer": {"name": "torch.optim.SGD"},
        "torch.optim.SGD": {"lr": 0.01, "momentum": 0.9},
        "scheduler": {"name": None},
    }

    model = ContextlessModel(config=config, data_sample=None)

    assert model.context == {}
    with pytest.raises(KeyError, match="outside of a verb run"):
        model.context["results_dir"]


@hyrax_model
class ContextWritingLoopback(nn.Module):
    """A loopback model that persists something of its own to the results dir.

    Defined at module scope so it survives the pickling that distributed training
    and the model registry perform.
    """

    def __init__(self, config, data_sample=None):
        from functools import partial

        super().__init__()
        self.config = config
        self.unused_module = nn.Linear(1, 1)
        self.batches_seen = 0

        def load(self, weight_file):
            """This model has no meaningful weights, so loading is a noop."""
            pass

        # Overridden this way rather than as a method because Torch's __init__
        # cleverness stomps a load defined in the usual fashion. Same trick as
        # HyraxLoopback.
        self.load = partial(load, self)

    def forward(self, x):
        """Return the input unchanged."""
        if isinstance(x, (tuple, list)):
            x, _ = x
        return x

    def train_batch(self, batch):
        """Training is a noop; just count the batch."""
        self.forward(batch)
        self.batches_seen += 1
        return {"loss": 0.0}

    def validate_batch(self, batch):
        """Validation is a noop."""
        self.forward(batch)
        return {"loss": 0.0}

    def infer_batch(self, batch):
        """Inference is just a forward pass."""
        return self.forward(batch)

    def test_batch(self, batch):
        """Testing is a noop; just count the batch."""
        self.forward(batch)
        self.batches_seen += 1
        return {"loss": 0.0}

    def train_post_epoch(self):
        """Persist something that does not fit the TensorBoard/MLflow paradigm."""
        self._write_notes()

    def test_post_epoch(self):
        """Same, at the end of the test pass."""
        self._write_notes()

    def _write_notes(self):
        """Write accumulated state into the results dir of the run underway.

        Read the context here rather than stashing a handle in __init__: this is
        the form that reports the right rank under distributed training, where a
        handle taken in the parent is pickled into each child as a stale copy.
        """
        context = get_context()
        rank = context["rank"]
        outfile = context["results_dir"] / f"my_notes_rank{rank}.json"
        with open(outfile, "w") as f:
            json.dump({"batches_seen": self.batches_seen, "verb": context["verb"]}, f)

    @staticmethod
    def prepare_inputs(data_dict):
        """Simple input prep, matching HyraxLoopback."""
        import numpy as np

        data = data_dict.get("data")
        image = data.get("image", np.array([], dtype=np.float32))
        label = data.get("label", np.array([], dtype=np.float32))
        return (image, label)


def test_model_writes_to_results_dir_during_train(loopback_hyrax):
    """End-to-end: a model uses the run context to save its own artifact into the
    train results directory."""
    h, _ = loopback_hyrax
    h.config["model"]["name"] = "ContextWritingLoopback"

    h.train()

    results_dir = find_most_recent_results_dir(h.config, "train")
    notes_file = results_dir / "my_notes_rank0.json"
    assert notes_file.exists()

    notes = json.loads(notes_file.read_text())
    assert notes["verb"] == "train"
    assert notes["batches_seen"] > 0


def test_model_writes_to_results_dir_during_test(loopback_hyrax):
    """The same pattern via test_post_epoch, which lands in the test results dir."""
    h, _ = loopback_hyrax
    h.config["model"]["name"] = "ContextWritingLoopback"
    h.config["test"]["model_weights_file"] = h.config["infer"]["model_weights_file"]

    h.test()

    results_dir = find_most_recent_results_dir(h.config, "test")
    notes_file = results_dir / "my_notes_rank0.json"
    assert notes_file.exists()

    notes = json.loads(notes_file.read_text())
    assert notes["verb"] == "test"
    assert notes["batches_seen"] > 0


def test_verbs_release_their_context(loopback_hyrax):
    """A verb's context lasts for its run and no longer.

    The in-run half is covered by test_model_writes_to_results_dir_during_train,
    where the model's notes land in the train results directory.
    """
    h, _ = loopback_hyrax

    h.train()
    assert get_context() == {}

    h.infer()
    assert get_context() == {}


def test_verb_without_a_results_dir_does_not_inherit_one(loopback_hyrax):
    """Verbs that never create a results directory used to see the previous run's,
    because nothing released it. The base class hands each verb its own context,
    which the wrapper also exposes on the instance as self.context."""
    from hyrax.verbs.model import Model

    h, _ = loopback_hyrax
    h.train()

    model_verb = Model(h.config)
    model_verb.run()

    assert model_verb.context["verb"] == "model"
    assert "results_dir" not in model_verb.context


def test_verb_run_signature_survives_the_context_wrapper():
    """Hyrax.__getattr__ hands verb.run straight to notebook users, who rely on it
    for help text and completion, so the wrapper must not flatten it to (*args)."""
    from hyrax.verbs.lookup import Lookup

    parameters = inspect.signature(Lookup.run).parameters

    assert "id" in parameters
    assert "results_dir" in parameters
    assert Lookup.run.__doc__ is not None


def test_vector_db_keeps_the_context_of_its_own_run(loopback_inferred_hyrax):
    """A vector database reads its directory from the run context and holds on to
    that context.

    database_connection hands the database object back to the user for interactive
    querying, and ChromaDB re-reads this path at query time to spawn its worker
    processes. Because each run gets its own context object, the one the database
    holds keeps pointing at the database directory however many verbs run later.
    """
    h, _, inference_results = loopback_inferred_hyrax
    h.config["vector_db"]["name"] = "chromadb"

    vdb_path = Path(h.config["general"]["results_dir"]).resolve()
    h.save_to_database(output_dir=vdb_path)
    db = h.database_connection(database_dir=vdb_path)

    assert db.results_dir == vdb_path

    # Run another verb, which gets its own context and then releases it.
    h.infer()
    assert get_context() == {}

    # The held connection must be unaffected and still able to query.
    assert db.results_dir == vdb_path
    an_id = list(inference_results.ids())[0]
    assert an_id in db.get_by_id(an_id)
