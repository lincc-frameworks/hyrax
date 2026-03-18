"""Tests for verb-time data_request validation.

Verbs validate their data_request configuration at instantiation time, restricted
to the groups they actually use (REQUIRED_DATA_GROUPS + OPTIONAL_DATA_GROUPS).
This prevents unrelated groups from causing false validation failures (issue #787).
"""

import pytest

from hyrax.config_utils import ConfigManager
from hyrax.verbs import Infer, Test, Train
from hyrax.verbs.verb_registry import Verb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(data_request: dict) -> dict:
    """Return a minimal config dict with the given data_request."""
    cm = ConfigManager()
    cm.set_config("data_request", data_request)
    return cm.config


def _make_single_group(group: str, data_location: str = "/tmp/data", **kwargs) -> dict:
    """Return a data_request dict with a single named group."""
    cfg = {
        "dataset_class": "HyraxRandomDataset",
        "data_location": data_location,
        "primary_id_field": "id",
    }
    cfg.update(kwargs)
    return {group: cfg}


# ---------------------------------------------------------------------------
# Verb base class: default DATA_GROUPS
# ---------------------------------------------------------------------------


def test_verb_default_data_groups_are_empty():
    """Verb base class defines empty REQUIRED_DATA_GROUPS and OPTIONAL_DATA_GROUPS."""
    assert Verb.REQUIRED_DATA_GROUPS == ()
    assert Verb.OPTIONAL_DATA_GROUPS == ()


# ---------------------------------------------------------------------------
# Verbs without DATA_GROUPS skip validation
# ---------------------------------------------------------------------------


def test_verb_without_data_groups_skips_validation():
    """A verb with no DATA_GROUPS does not validate data_request at all."""

    class NoGroupsVerb(Verb):
        """Minimal test verb with no data groups."""

        cli_name = "no_groups"
        add_parser_kwargs = {}

        @staticmethod
        def setup_parser(parser):
            """No parser setup needed."""

        def run_cli(self, args=None):
            """No CLI implementation."""

        def run(self):
            """No run implementation."""

    # Even a structurally valid-looking config that wouldn't pass validation should not raise
    # because the verb skips validation entirely when DATA_GROUPS are empty.
    cfg = ConfigManager().config
    cfg["data_request"] = {
        "some_group": {"dataset_class": "HyraxRandomDataset", "data_location": "/tmp/data"}
    }
    verb = NoGroupsVerb(cfg)
    assert verb is not None


def test_verb_without_data_request_config_skips_validation():
    """A verb with DATA_GROUPS but no data_request in config skips validation."""
    cm = ConfigManager()
    # Ensure neither data_request nor model_inputs is present
    cm.config.pop("data_request", None)
    cm.config.pop("model_inputs", None)
    assert "data_request" not in cm.config
    Train(cm.config)  # should not raise


# ---------------------------------------------------------------------------
# Required groups validation
# ---------------------------------------------------------------------------


def test_missing_required_group_raises():
    """Instantiating Train without 'train' group in data_request raises RuntimeError."""
    config = _make_config(_make_single_group("infer"))
    with pytest.raises(RuntimeError, match="requires dataset group"):
        Train(config)


def test_required_group_present_passes():
    """Train with 'train' group present instantiates without error."""
    config = _make_config(_make_single_group("train"))
    Train(config)  # should not raise


def test_infer_requires_infer_group():
    """Infer verb requires 'infer' group."""
    config = _make_config(_make_single_group("train"))
    with pytest.raises(RuntimeError, match="requires dataset group"):
        Infer(config)


def test_test_requires_test_group():
    """Test verb requires 'test' group."""
    config = _make_config(_make_single_group("train"))
    with pytest.raises(RuntimeError, match="requires dataset group"):
        Test(config)


# ---------------------------------------------------------------------------
# Cross-group validation — only active groups are checked (issue #787)
# ---------------------------------------------------------------------------


def test_issue_787_train_validate_with_infer_no_split():
    """Core issue-787 scenario: train+validate share split_fraction, infer does not.

    The Train verb only looks at 'train' and 'validate', so the absence of
    split_fraction on 'infer' must not cause a validation failure.
    """
    data_request = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.8,
        },
        "validate": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.2,
        },
        "infer": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            # No split_fraction — used for full-dataset inference
        },
    }
    config = _make_config(data_request)
    # Train should only see train+validate — passes
    Train(config)
    # Infer should only see infer — passes
    Infer(config)


def test_cross_group_split_fraction_exceeds_one_raises():
    """If the active groups sum split_fraction > 1.0, RuntimeError is raised."""
    data_request = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.7,
        },
        "validate": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.5,
        },
    }
    config = _make_config(data_request)
    with pytest.raises(RuntimeError, match="exceeds 1.0"):
        Train(config)


def test_cross_group_split_fraction_consistency_violation_raises():
    """If one group sets split_fraction and another (active) group does not, raise."""
    data_request = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.7,
        },
        "validate": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            # Missing split_fraction for a shared location — consistency violation
        },
    }
    config = _make_config(data_request)
    with pytest.raises(RuntimeError, match="split_fraction"):
        Train(config)


def test_optional_group_included_in_active_set():
    """Optional groups (e.g. 'validate') are included in cross-group checks."""
    # 'validate' is optional for Train but still participates in cross-group validation
    data_request = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.7,
        },
        "validate": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.5,
        },
    }
    config = _make_config(data_request)
    # 0.7 + 0.5 > 1.0, Train should catch this
    with pytest.raises(RuntimeError, match="exceeds 1.0"):
        Train(config)


def test_inactive_groups_do_not_affect_validation():
    """Groups outside REQUIRED+OPTIONAL are ignored in cross-group checks."""
    data_request = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 0.8,
        },
        # 'infer' is outside Train's active groups — its absence of split_fraction
        # must not cause a consistency failure for Train
        "infer": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
        },
    }
    config = _make_config(data_request)
    Train(config)  # should not raise


# ---------------------------------------------------------------------------
# model_inputs backward compat
# ---------------------------------------------------------------------------


def test_model_inputs_fallback_validated():
    """validate_data_request falls back to 'model_inputs' when 'data_request' absent."""
    cm = ConfigManager()
    cm.config.pop("data_request", None)
    cm.set_config(
        "model_inputs",
        {
            "train": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/tmp/data",
                "primary_id_field": "id",
            }
        },
    )
    Train(cm.config)  # should not raise — 'train' group is present


def test_model_inputs_missing_required_group_raises():
    """Missing required group in model_inputs also raises RuntimeError."""
    cm = ConfigManager()
    cm.config.pop("data_request", None)
    cm.set_config(
        "model_inputs",
        {
            "infer": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/tmp/data",
                "primary_id_field": "id",
            }
        },
    )
    with pytest.raises(RuntimeError, match="requires dataset group"):
        Train(cm.config)
