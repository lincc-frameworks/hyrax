import logging
import os

import pytest

from hyrax.config_utils import ConfigDict, ConfigManager


def test_merge_configs():
    """Basic test to ensure that the merge_configs function will join two dictionaries
    correctly, meaning:
    1) The user_config values should override the default_config values.
    2) Values in the default_config that are not in the user_config should remain unchanged.
    3) Values in the user_config that are not in the default_config should be added.
    4) Nested dictionaries should be merged recursively.
    """
    default_config = {
        "a": 1,
        "b": 2,  # This tests case 2
        "c": {"d": 3, "e": 4},
    }

    user_config = {
        "a": 5,  # This tests case 1
        "c": {
            "d": 6  # This tests case 4
        },
        "f": 7,  # This tests case 3
    }

    expected = {"a": 5, "b": 2, "c": {"d": 6, "e": 4}, "f": 7}

    assert ConfigManager.merge_configs(default_config, user_config) == expected


def test_get_runtime_config():
    """Test that the get_runtime_config function will load the default and user defined
    runtime configuration files, merge them, and return the final configuration as a
    dictionary.
    """

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    expected = {
        "general": {"dev_mode": True},
        "train": {
            "model_name": "example_model",
            "model_class": "new_thing.cool_model.CoolModel",
            "model": {"weights_filename": "final_best.pth", "layers": 3},
        },
        "infer": {"batch_size": 8},
        "bespoke_table": {"key1": "value1", "key2": "value2"},
    }

    string_representation = """# this is the default config file
[general]
# set dev_mode to true when developing
# set to false for production use
dev_mode = true

[train]
model_name = "example_model" # Use a built-in Hyrax model
model_class = "new_thing.cool_model.CoolModel" # Use a custom model

[train.model]
weights_filename = "final_best.pth"
layers = 3


[infer]
batch_size = 8 # change batch size

[bespoke_table]
# this is a bespoke table
key1 = "value1"
key2 = "value2" # unlikely to modify
"""
    assert runtime_config == expected
    assert runtime_config.as_string() == string_representation


def test_validate_runtime_config(caplog):
    """Test that the validate_runtime_config function will log a warning
    if a user key is not defined in the default configuration dictionary.
    """

    default = {"general": {"dev_mode": False}, "train": {"model_name": "example_model"}}
    default_config = ConfigDict(default)

    user = {"general": {"dev_mode": False, "foo": "bar"}}
    user_config = ConfigDict(user)

    with caplog.at_level(logging.WARNING):
        ConfigManager._validate_runtime_config(user_config, default_config)

    assert "Runtime config contains key" in caplog.text


def test_validate_runtime_config_section(caplog):
    """Test that the validate_runtime_config function will log a warning
    if a user section name conflicts with a default configuration key.
    """

    default = {"general": {"dev_mode": False}, "train": {"model_name": "example_model"}}
    default_config = ConfigDict(default)

    user = {"general": {"dev_mode": {"b": 2}}}
    user_config = ConfigDict(user)

    with caplog.at_level(logging.WARNING):
        ConfigManager._validate_runtime_config(user_config, default_config)

    assert "Runtime config contains a section named 'dev_mode'" in caplog.text


def test_find_external_library_config_path_no_module():
    """Test that a ModuleNotFound error is raised when trying to import a
    non-existent module.
    """

    config = {"general": {"dev_mode": False}, "model": {"name": "foo.bar.model.Model"}}
    default_config = ConfigDict(config)

    with pytest.raises(ModuleNotFoundError) as excinfo:
        ConfigManager._find_external_library_default_config_paths(default_config)

    assert "Check installation" in str(excinfo.value)


def test_find_external_library_config_path_no_default_config(caplog):
    """Test that a warning is logged when a default configuration file is not found."""

    config = {"general": {"dev_mode": False}, "model": {"name": "toml.bar.model.Model"}}
    default_config = ConfigDict(config)

    with caplog.at_level(logging.WARNING):
        ConfigManager._find_external_library_default_config_paths(default_config)

    assert "default_config.toml" in caplog.text


def test_config_help(capsys):
    """Basic use case where config help function prints entire config"""

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    runtime_config.help()

    captured = capsys.readouterr()

    expected_output = """# this is the default config file
[general]
# set dev_mode to true when developing
# set to false for production use
dev_mode = true

[train]
model_name = "example_model" # Use a built-in Hyrax model
model_class = "new_thing.cool_model.CoolModel" # Use a custom model

[train.model]
weights_filename = "final_best.pth"
layers = 3


[infer]
batch_size = 8 # change batch size

[bespoke_table]
# this is a bespoke table
key1 = "value1"
key2 = "value2" # unlikely to modify
"""

    assert expected_output in captured.out


def test_config_help_specific_table(capsys):
    """Basic use case where config help function prints one table"""

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    runtime_config.help("bespoke_table")

    captured = capsys.readouterr()

    expected_output = """[bespoke_table]
# this is a bespoke table
key1 = "value1"
key2 = "value2" # unlikely to modify
"""

    assert expected_output in captured.out


def test_config_help_table_and_key(capsys):
    """Basic use case where config help function prints table when provided with
    a table and key."""

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    runtime_config.help("train", "model_name")

    captured = capsys.readouterr()

    expected_output = """[train]
model_name = "example_model" # Use a built-in Hyrax model
model_class = "new_thing.cool_model.CoolModel" # Use a custom model
"""

    assert expected_output in captured.out


def test_config_help_non_existant_table_or_key(capsys):
    """Basic use case where config help function prints error message when table
    or key isn't present"""

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    non_table_name = "non_existant_table"
    runtime_config.help(non_table_name)

    captured = capsys.readouterr()

    expected_output = f"Could not find '{non_table_name}' in the config"

    assert expected_output in captured.out


def test_config_help_real_table_non_existant_key(capsys):
    """User requests help for real table, but non-existant key"""

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    runtime_config.help("general", "non_existant_key")

    captured = capsys.readouterr()

    expected_output = "Cannot find ['general']['non_existant_key'] in the current configuration."

    assert expected_output in captured.out


def test_config_help_key_in_multiple_tables(capsys):
    """User requests help for a key that is present in more than one table"""

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config_repeated_keys.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    runtime_config.help("name")

    captured = capsys.readouterr()

    expected_output = """Found 'name' in the following config sections: [model], [loss], [optimizer]
[model]
name = "resnet"
layers = 3


[loss]
name = "cross_entropy"


[optimizer]
name = "adam"
"""

    assert expected_output in captured.out


def test_config_help_too_many_args(capsys):
    """User requests help for with >2 input args"""

    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    config_manager = ConfigManager(
        runtime_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_user_config.toml")
        ),
        default_config_filepath=os.path.abspath(
            os.path.join(this_file_dir, "./test_data/test_default_config.toml")
        ),
    )

    runtime_config = config_manager.config

    runtime_config.help("general", "dev_mode", "extra_arg")

    captured = capsys.readouterr()

    expected_output = """Too many arguments provided. Expecting 0, 1, or 2 arguments.
Usage: config.help(['table_name'|'key_name']), config.help('table_name', 'key_name')"""

    assert expected_output in captured.out


def test_resolve_runtime_config_non_existent_file():
    """Test that resolve_runtime_config raises FileNotFoundError for non-existent config files."""
    non_existent_file = "/path/to/non/existent/config.toml"

    with pytest.raises(FileNotFoundError, match=f"Cannot find config file {non_existent_file}"):
        ConfigManager.resolve_runtime_config(non_existent_file)


def test_resolve_runtime_config_valid_file():
    """Test that resolve_runtime_config works correctly with an existing file."""
    # Use a file that we know exists - the default config
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    existing_config = os.path.abspath(os.path.join(this_file_dir, "./test_data/test_user_config.toml"))

    # This should not raise an exception
    result = ConfigManager.resolve_runtime_config(existing_config)
    assert str(result) == existing_config


def test_resolve_runtime_config_none():
    """Test that resolve_runtime_config works correctly with None (fallback behavior)."""
    # This should not raise an exception and should return the default config
    result = ConfigManager.resolve_runtime_config(None)
    # The result should be one of the default paths
    assert result is not None
