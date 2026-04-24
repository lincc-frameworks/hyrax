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
    """Return a data_request dict with a single named group.

    The dataset source is nested under the friendly name ``"data"`` as required
    by the schema (a friendly name must always be explicitly provided).
    """
    cfg = {
        "dataset_class": "HyraxRandomDataset",
        "data_location": data_location,
        "primary_id_field": "id",
    }
    cfg.update(kwargs)
    return {group: {"data": cfg}}


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
    # Ensure data_request is not present
    cm.config.pop("data_request", None)
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
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/tmp/data",
                "primary_id_field": "id",
                "split_fraction": 0.8,
            }
        },
        "validate": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/tmp/data",
                "primary_id_field": "id",
                "split_fraction": 0.2,
            }
        },
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/tmp/data",
                "primary_id_field": "id",
                # No split_fraction — used for full-dataset inference
            }
        },
    }
    config = _make_config(data_request)
    # Train should only see train+validate — passes
    Train(config)
    # Infer should only see infer — passes
    Infer(config)


def test_inactive_groups_do_not_affect_validation():
    """Groups outside REQUIRED+OPTIONAL are ignored in cross-group checks."""
    data_request = {
        "train": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/tmp/data",
                "primary_id_field": "id",
                "split_fraction": 0.8,
            }
        },
        # 'infer' is outside Train's active groups — its absence of split_fraction
        # must not cause a consistency failure for Train
        "infer": {
            "data": {
                "dataset_class": "HyraxRandomDataset",
                "data_location": "/tmp/data",
                "primary_id_field": "id",
            }
        },
    }
    config = _make_config(data_request)
    Train(config)  # should not raise


# ---------------------------------------------------------------------------
# Robustness: non-mapping and structurally invalid data_request configs
# ---------------------------------------------------------------------------


def test_non_mapping_data_request_raises():
    """A non-mapping data_request config raises RuntimeError with a clear message.

    ConfigManager.set_config() stores invalid values as-is after a ValidationError.
    When a verb is later instantiated it must surface the problem clearly.
    """
    cm = ConfigManager()
    # Bypass set_config to inject a non-mapping value directly.
    cm.config["data_request"] = "this_is_not_a_dict"
    with pytest.raises(RuntimeError, match="non-mapping"):
        Train(cm.config)


def test_structurally_invalid_data_request_raises():
    """A mapping data_request that fails pydantic validation raises RuntimeError.

    ConfigManager.set_config() may store a structurally invalid config as-is
    (logging a warning).  Verb instantiation must surface the pydantic error
    as a clear RuntimeError instead of silently skipping cross-group validation.
    """
    cm = ConfigManager()
    # Flat dict without a friendly name — fails schema validation with a
    # "friendly name" error; inject directly to simulate set_config storing
    # the bad config as-is after its own ValidationError catch.
    cm.config["data_request"] = {
        "train": {
            "dataset_class": "HyraxRandomDataset",
            "data_location": "/tmp/data",
            "primary_id_field": "id",
            "split_fraction": 1.5,  # also out of range (le=1.0)
        }
    }
    with pytest.raises(RuntimeError, match="Invalid data_request"):
        Train(cm.config)


# ---------------------------------------------------------------------------
# data_request required-group enforcement
# ---------------------------------------------------------------------------


def test_data_request_missing_required_group_raises():
    """Missing required group in data_request also raises RuntimeError."""
    cm = ConfigManager()
    cm.config.pop("data_request", None)
    cm.set_config(
        "data_request",
        {
            "infer": {
                "data": {
                    "dataset_class": "HyraxRandomDataset",
                    "data_location": "/tmp/data",
                    "primary_id_field": "id",
                }
            }
        },
    )
    with pytest.raises(RuntimeError, match="requires dataset group"):
        Train(cm.config)
