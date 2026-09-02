import inspect
from pathlib import Path

import pytest

from hyrax.context import (
    RunContext,
    clear_context,
    get_context,
    run_context,
    update_context,
    use_context,
)


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
