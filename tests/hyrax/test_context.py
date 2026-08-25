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
    init_context,
    update_context,
)
from hyrax.models.model_registry import hyrax_model


@pytest.fixture(autouse=True)
def clean_context():
    """The run context is process-wide, so make sure each test starts and ends clean."""
    clear_context()
    yield
    clear_context()


def test_context_is_a_dict():
    """The context must be an ordinary dict so it can be passed to anything that
    expects one - vector databases, the ONNX exporter, ** unpacking, pickling."""
    context = get_context()
    assert isinstance(context, dict)
    assert isinstance(context, RunContext)
    assert context == {}


def test_handle_taken_before_init_sees_values():
    """The late-binding guarantee. A model may take a handle in __init__ before the
    verb has populated the context; that handle must reflect the values when they
    arrive. This is why init_context mutates in place rather than rebinding."""
    early_handle = get_context()
    assert early_handle == {}

    init_context("/some/results/dir", "train")

    assert early_handle is get_context()
    assert early_handle["verb"] == "train"
    assert early_handle["results_dir"] == Path("/some/results/dir")


def test_init_context_populates_expected_keys():
    """init_context fills in the documented key set."""
    context = init_context("/some/results/dir", "infer")

    assert context["results_dir"] == Path("/some/results/dir")
    assert context["verb"] == "infer"
    assert context["rank"] == 0
    assert context["world_size"] == 1


def test_init_context_coerces_results_dir_to_path():
    """save_to_database used to pass a str; model_exporters uses the `/` operator."""
    context = init_context("/some/results/dir", "vector-db")
    assert isinstance(context["results_dir"], Path)

    context = init_context(Path("/some/results/dir"), "vector-db")
    assert isinstance(context["results_dir"], Path)


def test_init_context_accepts_extra_keys():
    """The to_onnx verb adds ml_framework this way."""
    context = init_context("/some/results/dir", "onnx", ml_framework="pytorch")
    assert context["ml_framework"] == "pytorch"


def test_init_context_clears_previous_run():
    """Starting a new run must not leak the previous run's keys."""
    init_context("/first/run", "train")
    update_context(user_key="stashed by a model")

    init_context("/second/run", "infer")

    assert get_context()["results_dir"] == Path("/second/run")
    assert get_context()["verb"] == "infer"
    assert "user_key" not in get_context()


def test_update_context_preserves_existing_keys():
    """train.py's _training uses update rather than init so that anything a model
    stashed during setup_model survives on the single-process path."""
    init_context("/some/run", "train")
    update_context(user_key="stashed by a model")

    update_context(results_dir=Path("/some/run"), verb="train", rank=1, world_size=4)

    assert get_context()["user_key"] == "stashed by a model"
    assert get_context()["rank"] == 1
    assert get_context()["world_size"] == 4


def test_update_context_coerces_results_dir_to_path():
    """Coercion applies on the update path too, not just init."""
    update_context(results_dir="/some/run")
    assert get_context()["results_dir"] == Path("/some/run")


def test_missing_key_raises_helpful_error():
    """Hyrax users are scientists, not software engineers. A bare KeyError from a
    dict lookup would not tell them why the context was empty."""
    init_context("/some/run", "train")

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
    """clear_context empties the context without replacing the object."""
    init_context("/some/run", "train")
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
        # A handle taken here, before the verb has necessarily filled the context.
        self.context = get_context()
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
        """Write accumulated state into whichever run's results dir is current."""
        rank = self.context["rank"]
        outfile = self.context["results_dir"] / f"my_notes_rank{rank}.json"
        with open(outfile, "w") as f:
            json.dump({"batches_seen": self.batches_seen, "verb": self.context["verb"]}, f)

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


def test_verbs_populate_the_context(loopback_hyrax):
    """Each verb should leave the context pointing at its own results directory."""
    h, _ = loopback_hyrax

    h.train()
    assert get_context()["verb"] == "train"
    assert get_context()["results_dir"] == find_most_recent_results_dir(h.config, "train")

    h.infer()
    assert get_context()["verb"] == "infer"
    assert get_context()["results_dir"] == find_most_recent_results_dir(h.config, "infer")


def test_vector_db_snapshots_its_directory(loopback_inferred_hyrax):
    """A vector database reads its directory from the run context, but must snapshot
    it rather than hold the live context.

    database_connection hands the database object back to the user for interactive
    querying, and ChromaDB re-reads this path at query time to spawn its worker
    processes. If it followed the live run context, a held connection would start
    looking for the database inside whatever results directory the next verb made.
    """
    h, _, inference_results = loopback_inferred_hyrax
    h.config["vector_db"]["name"] = "chromadb"

    vdb_path = Path(h.config["general"]["results_dir"]).resolve()
    h.save_to_database(output_dir=vdb_path)
    db = h.database_connection(database_dir=vdb_path)

    assert db.results_dir == vdb_path

    # Move the run context on to a different directory.
    h.infer()
    assert get_context()["results_dir"] != vdb_path

    # The held connection must be unaffected and still able to query.
    assert db.results_dir == vdb_path
    an_id = list(inference_results.ids())[0]
    assert an_id in db.get_by_id(an_id)
