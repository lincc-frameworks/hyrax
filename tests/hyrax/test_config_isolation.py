"""Tests that verb dispatch in Hyrax isolates config via deepcopy.

Regression tests for https://github.com/lincc-frameworks/hyrax/issues/703
"""

from copy import deepcopy

import hyrax


def test_sequential_train_infer_uses_fresh_config(loopback_hyrax):
    """The exact scenario from issue #703: train→infer→train→infer.

    After each cycle, h.config must remain unchanged so the second
    infer() doesn't silently reuse weights from the first training run.
    """
    h, _ = loopback_hyrax

    # Reset model_weights_file to False (the TOML sentinel for None) so that
    # infer() must auto-detect weights from the most recent train run.
    # This is the code path where the bug manifested: the auto-detected path
    # was written back into h.config, causing the second infer to reuse the
    # first run's weights.
    h.config["infer"]["model_weights_file"] = False

    config_snapshot = deepcopy(h.config)

    # First cycle
    h.train()
    h.infer()
    assert h.config == config_snapshot

    # Second cycle — before the fix, infer would silently reuse first-run weights
    h.train()
    h.infer()
    assert h.config == config_snapshot


def test_prepare_does_not_mutate_hyrax_config(tmp_path):
    """Calling h.prepare() must not modify h.config."""
    h = hyrax.Hyrax()
    h.config["model"]["name"] = "HyraxLoopback"
    h.config["general"]["results_dir"] = str(tmp_path)
    h.config["general"]["dev_mode"] = True
    h.config["model_inputs"] = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data"),
                "primary_id_field": "object_id",
            },
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": str(tmp_path / "data_infer"),
                "primary_id_field": "object_id",
            },
        },
    }
    h.config["data_set"]["HyraxRandomDataset"]["size"] = 10
    h.config["data_set"]["HyraxRandomDataset"]["seed"] = 0
    h.config["data_set"]["HyraxRandomDataset"]["shape"] = [2, 3]

    config_snapshot = deepcopy(h.config)

    h.prepare()

    assert h.config == config_snapshot


def test_set_config_visible_to_subsequent_verb(loopback_hyrax):
    """Config changes via set_config() must be visible to the next verb call.

    This guards against the deepcopy happening too early (e.g. at
    construction time) rather than at call time.
    """
    h, _ = loopback_hyrax

    h.set_config("train.epochs", 2)
    assert h.config["train"]["epochs"] == 2

    # train() should see the updated value — if deepcopy were stale
    # the verb would still see epochs=1 and the config saved in
    # results would disagree.
    h.train()

    # Config should still reflect the user's setting after the verb returns
    assert h.config["train"]["epochs"] == 2
