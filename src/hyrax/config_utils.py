import base64
import copy
import datetime
import importlib
import logging
import random
import re
from importlib import util as importlib_util
from pathlib import Path
from typing import Any, Optional, Union

import tomlkit
from tomlkit.toml_document import TOMLDocument

DEFAULT_CONFIG_FILEPATH = Path(__file__).parent.resolve() / "hyrax_default_config.toml"
DEFAULT_USER_CONFIG_FILEPATH = Path.cwd() / "hyrax_config.toml"
# There are only a couple of configuration keys where we would expect to find an
# external library string, so we specify those here.
KEYS_WITH_EXTERNAL_LIBS = ["name", "dataset_class"]

logger = logging.getLogger(__name__)


class ConfigDict(dict):
    """The purpose of this class is to ensure key errors on config dictionaries return something helpful.
    and to discourage mutation actions on config dictionaries that should not happen at runtime.
    """

    # TODO: Should there be some sort of "bake" method which occurs after config processing, and
    # percolates down to nested ConfigDicts and prevents __setitem__ and other mutations of dictionary
    # values? i.e. a method to make a config dictionary fully immutable (or very difficult/annoying to
    # mutate) before we pass control to possibly external module code that is relying on the dictionary
    # to be static throughout the run.
    #
    # Note that we currently modify ConfigDict objects in HSCDataSet as a matter of course
    # This is to serialize the config used to create them, so it can be kept around in InferenceDatasets
    # and re-constituted for visualization purposes

    __slots__ = ()  # we don't need __dict__ on this object at all.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace all dictionary keys with values recursively.
        for key, val in self.items():
            if isinstance(val, dict) and not isinstance(val, ConfigDict):
                self[key] = ConfigDict(val)

    def __missing__(self, key):
        msg = f"Accessed configuration key/section {key} which has not been defined. "
        msg += f"All configuration keys and sections must be defined in {DEFAULT_CONFIG_FILEPATH}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    def get(self, key, default=None):
        """Nonfunctional stub of dict.get() which errors always. Overriding this
        prevents the possibility of defining default values in code that conflict
        with default values defined in the config file."""
        msg = f"ConfigDict.get({key},{default}) called. "
        msg += "Please index config dictionaries with [] or __getitem__() only. "
        msg += f"Configuration keys and sections must be defined in {DEFAULT_CONFIG_FILEPATH}"
        logger.fatal(msg)
        raise RuntimeError(msg)

    def __delitem__(self, key):
        raise RuntimeError("Removing keys or sections from a ConfigDict using del is not supported")

    def pop(self, key, default):
        """Nonfunctional stub of dict.pop() which errors always"""
        raise RuntimeError("Removing keys or sections from a ConfigDict using pop() is not supported")

    def popitem(self):
        """Nonfunctional stub of dict.popitem() which errors always"""
        raise RuntimeError("Removing keys or sections from a ConfigDict using popitem() is not supported")

    def clear(self):
        """Nonfunctional stub of dict.clear() which errors always"""
        raise RuntimeError("Removing keys or sections from a ConfigDict using clear() is not supported")


def config_help(config: TOMLDocument, *args):
    """
    A simple config help function. It's a bit difficult to parse through
    the Tomlkit Table to print just one item such that it would include the comments
    preceding it.

    For now, we support the following cases, and generally print out the entire
    table for the given key.

    Cases:
    - if no args, prints the whole config.
    - if args[0] is a table name, print the whole table
    - if args[0] is not a table, assume it's a key and search
    -- print each one of the tables that it is found in.

    Parameters
    ----------
    config : TOMLDocument
        A configuration dictionary that will be used to search for specified tables
        and keys.

    args : str
        A variable number of string arguments that specify the table name or key
        to search for in the configuration dictionary.
    """

    # Get the config as a dictionary
    config_dict = config.value

    # No tables provided, print the whole config
    if not args:
        print(config.as_string())

    # Table name provided as args[0], print that config table
    if len(args) == 1 and args[0] in config_dict:
        print(f"[{args[0]}]")
        print(config[args[0]].as_string())

    # One arg provided, but it's not a table name.
    # Assume it's a config key and search the config for it.
    # Print each table that it is found in.
    if len(args) == 1 and args[0] not in config_dict:
        matching = find_keys(config_dict, args[0])
        if len(matching):
            tables = [m.split(".")[0] for m in matching]
            print(f"Found '{args[0]}' in the following config sections: [{'], ['.join(tables)}]")
            for t in tables:
                config_help(config, t)
        else:
            print(f"Could not find '{args[0]}' in the config")

    if len(args) == 2:
        if args[0] in config_dict and args[1] in config_dict[args[0]]:
            print(f"[{args[0]}]")
            print(config[args[0]].as_string())
        else:
            print(f"Cannot find ['{args[0]}']['{args[1]}'] in the current configuration.")

    if len(args) > 2:
        print("Too many arguments provided. Expecting 0, 1, or 2 arguments.")
        print("Usage: config.help(['table_name'|'key_name']), config.help('table_name', 'key_name')")


def find_keys(config: dict[str, Any], key_name: str):
    """
    Recursively find all keys in a nested dictionary that match the given key name.

    Parameters
    ----------
    config : dict
        The nested dictionary to search.
    key_name : str
        The name of the key to find.

    Returns
    -------
    list
        A list of matching keys.
    """
    matching_keys = []

    def _find_keys(d, parent_key=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if k == key_name:
                    matching_keys.append((parent_key + "." + k).strip("."))
                _find_keys(v, parent_key + "." + k)

    _find_keys(config)
    return matching_keys


# Here we patch the TOMLDocument functions to use the ConfigDict functions so that
# we get the behavior that we desire - i.e. errors on missing default keys, no
# use of config.get(..., <default>), etc.
TOMLDocument.__missing__ = ConfigDict.__missing__  # type: ignore[attr-defined]
TOMLDocument.get = ConfigDict.get  # type: ignore[assignment, method-assign]
TOMLDocument.__delitem__ = ConfigDict.__delitem__  # type: ignore[assignment, method-assign]
TOMLDocument.pop = ConfigDict.pop  # type: ignore[assignment, method-assign]
TOMLDocument.popitem = ConfigDict.popitem  # type: ignore[assignment, method-assign]
TOMLDocument.clear = ConfigDict.clear  # type: ignore[assignment, method-assign]
TOMLDocument.help = config_help  # type: ignore


class ConfigManager:
    """A class to manage the runtime configuration for a Hyrax object. This class
    will contain all the logic and methods for reading, merging, and validating
    the runtime configuration.
    """

    # True when we are called from a test, so we maintain isolation from any
    # user config that may be in the cwd of the running test process
    _called_from_test = False

    """
    Hardcoded set of config keys which we know to contain paths, and we resolve to global paths
    during initialization in ConfigManager._resolve_config_paths().
    """
    PATH_CONFIG_KEYS = [
        # TODO: external library config defaults
        # However we define config defaults from external libraries ought allow them to designate config keys
        # which contain relative paths. ultimately these should end up on the list and be resolved.
        ["data_set", "filter_catalog"],
        ["general", "data_dir"],
    ]

    def __init__(
        self,
        runtime_config_filepath: Optional[Union[Path, str]] = None,
        default_config_filepath: Union[Path, str] = DEFAULT_CONFIG_FILEPATH,
    ):
        self.hyrax_default_config: TOMLDocument = ConfigManager.read_runtime_config(default_config_filepath)

        self.runtime_config_filepath = ConfigManager.resolve_runtime_config(runtime_config_filepath)
        if self.runtime_config_filepath is DEFAULT_CONFIG_FILEPATH:
            self.user_specific_config = TOMLDocument()
        else:
            self.user_specific_config = ConfigManager.read_runtime_config(self.runtime_config_filepath)

        self.config = self._render_config(self.user_specific_config, self.hyrax_default_config)
        self.original_config = copy.deepcopy(self.config)

    @staticmethod
    def _render_config(
        user_specific_config: TOMLDocument = None,
        hyrax_default_config: TOMLDocument = None,
    ):
        user_specific_config = user_specific_config if user_specific_config is not None else TOMLDocument()
        hyrax_default_config = hyrax_default_config if hyrax_default_config is not None else TOMLDocument()

        external_library_config_paths = set()
        external_library_config_paths |= ConfigManager._find_external_library_default_config_paths(
            user_specific_config
        )

        # 1) merge all the external library config dictionaries together
        external_default_config = ConfigManager.merge_external_default_configs(external_library_config_paths)

        # 2) merge the external library configs on top of the hyrax defaults
        overall_default_config = ConfigManager.merge_default_configs(
            hyrax_default_config, external_default_config
        )

        # 3) merge the user config on top of the overall defaults
        config = ConfigManager.merge_configs(overall_default_config, user_specific_config)

        ConfigManager._resolve_config_paths(config)
        if not config["general"]["dev_mode"]:
            ConfigManager._validate_runtime_config(config, overall_default_config)

        return config

    def set_config(self, key: str, value: Any):
        """Set a config value at runtime. This modifies the in-memory config object.
        Once the configuration is updated, the entire config is re-rendered to
        ensure that any requested external library default configs are incorporated.

        Parameters
        ----------
        key : str
            The dotted key to set, e.g. "model.name"

        value : Any
            The value to set the key to.
        """
        keys = key.split(".")
        d = self.config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

        self.config = self._render_config(self.config, self.original_config)
        self.original_config = copy.deepcopy(self.config)

    @staticmethod
    def read_runtime_config(config_filepath: Union[Path, str] = DEFAULT_CONFIG_FILEPATH) -> TOMLDocument:
        """Read a single toml file and return a ConfigDict

        Parameters
        ----------
        config_filepath : Union[Path, str], optional
            The path to the config file, by default DEFAULT_CONFIG_FILEPATH

        Returns
        -------
        TOMLDocument
            The contents of the toml file as a tomlkit.TOMLDocument
        """
        config_filepath = Path(config_filepath)
        parsed_dict: TOMLDocument = TOMLDocument()
        if config_filepath.exists():
            with open(config_filepath, "r") as f:
                parsed_dict = tomlkit.load(f)

        return parsed_dict

    @staticmethod
    def _find_external_library_default_config_paths(runtime_config: dict) -> set:
        """Search for external libraries in the runtime configuration and gather the
        libpath specifications so that we can load the default configs for the libraries.

        Parameters
        ----------
        runtime_config : dict
            The runtime configuration as a tomlkit.TOMLDocument.
        Returns
        -------
        set
            A tuple containing the default configuration Paths for the external
            libraries that are requested in the users configuration file.
        """

        default_config_paths = set()
        for key, value in runtime_config.items():
            if isinstance(value, dict):
                default_config_paths |= ConfigManager._find_external_library_default_config_paths(value)
            else:
                if key in KEYS_WITH_EXTERNAL_LIBS and "." in value:
                    external_library = value.split(".")[0]
                    if importlib_util.find_spec(external_library) is not None:
                        try:
                            lib = importlib.import_module(external_library)
                            if lib.__file__ is None:
                                raise RuntimeError()
                            lib_default_config_path = Path(lib.__file__).parent / "default_config.toml"
                            if lib_default_config_path.exists():
                                default_config_paths.add(lib_default_config_path)
                            else:
                                logger.warning(f"Cannot find default_config.toml for {value}.")
                        except ModuleNotFoundError:
                            logger.error(
                                f"External library {lib} not found. Please install it before running."
                            )
                            raise
                    else:
                        raise ModuleNotFoundError(
                            f"External library {external_library} not found. Check installation."
                        )

        return default_config_paths

    @staticmethod
    def merge_external_default_configs(external_default_config_paths):
        """Merge the default configurations from external libraries into the overall
        default configuration.

        Parameters
        ----------
        external_default_config_paths : set
            A set containing the default configuration Paths for the external
            libraries that are requested in the users configuration file.

        Returns
        -------
        dict
            The merged overall default configuration including the external library defaults.
        """
        overall_default_config = TOMLDocument()
        # Merge all external library default configurations first
        for path in external_default_config_paths:
            logger.info(f"Merging external default config from {path}")
            external_library_config = ConfigManager.read_runtime_config(path)
            overall_default_config = ConfigManager.merge_configs(
                overall_default_config, external_library_config
            )

        return overall_default_config

    @staticmethod
    def merge_default_configs(hyrax_defaults, external_defaults):
        """Merge the default configurations of external libraries on top of the
        Hyrax default configuration.

        Parameters
        ----------
        hyrax_defaults : dict
            The default configuration from hyrax.
        external_defaults : dict
            The default configuration from external libraries.

        Returns
        -------
        dict
            The merged overall default configuration including the external library defaults.
        """
        return ConfigManager.merge_configs(hyrax_defaults, external_defaults)

    @staticmethod
    def merge_configs(base_config: dict, overriding_config: dict) -> dict:
        """Merge two ConfigDicts with the overriding_config values overriding
        the default_config values.

        Parameters
        ----------
        base_config : dict
            The base configuration with keys that may be overridden by the
            overriding_config.
        overriding_config : dict
            The new configuration values that will override the values in base_config.

        Returns
        -------
        dict
            The merged configuration.
        """

        final_config = base_config.copy()
        for k, v in overriding_config.items():
            if k in final_config and isinstance(final_config[k], dict) and isinstance(v, dict):
                final_config[k] = ConfigManager.merge_configs(base_config[k], v)
            else:
                final_config[k] = v

        return final_config

    @staticmethod
    def _validate_runtime_config(runtime_config: dict, default_config: dict):
        """Recursive helper to check that all keys in runtime_config have a default
        in the merged default_config.

        The two arguments passed in must represent the same nesting level of the
        runtime config and all default config parameters respectively.

        Parameters
        ----------
        runtime_config : ConfigDict
            Nested config dictionary representing the runtime config.
        default_config : ConfigDict
            Nested config dictionary representing the defaults

        Raises
        ------
        RuntimeError
            Raised if any config that exists in the runtime config does not have a default defined in
            default_config
        """
        for key in runtime_config:
            if key not in default_config:
                msg = f"Runtime config contains key or section '{key}' which has no default defined. "
                msg += f"All configuration keys and sections must be defined in {DEFAULT_CONFIG_FILEPATH}"
                logger.warning(msg)
                continue

            if isinstance(runtime_config[key], dict):
                if not isinstance(default_config[key], dict):
                    msg = f"Runtime config contains a section named '{key}' which is the name of a "
                    msg += "value in the default config. Please choose another name for this section."
                    logger.warning(msg)
                    continue
                ConfigManager._validate_runtime_config(runtime_config[key], default_config[key])

    @staticmethod
    def _resolve_config_paths(runtime_config: dict) -> None:
        """Convert all paths in a runtime config to global paths in the current environment.
        Uses the hardcoded list of paths in ConfigManager.PATH_CONFIG_KEYS

        This mutates the config dictionary passed.

        Parameters
        ----------
        runtime_config : dict
           Current runtime config nested dictionary
        """
        for key_spec in ConfigManager.PATH_CONFIG_KEYS:
            # Recursively look up a list of keys.
            current_dict = runtime_config
            current_key = None
            current_val = None
            # At the end of each loop current_* are always the dict, key, and value of the
            # last lookup.
            for key in key_spec:
                current_key = key
                try:
                    current_val = current_dict[key]
                except KeyError:
                    break

                if isinstance(current_val, dict):
                    current_dict = current_val

            # On the non-break end of the loop we do path resolution, preserving falsy values
            # as false.
            else:
                new_val = str(Path(current_val).expanduser().resolve()) if current_val else False
                current_dict[current_key] = new_val

    @staticmethod
    def resolve_runtime_config(runtime_config_filepath: Union[Path, str, None] = None) -> Path:
        """Resolve a user-supplied runtime config to where we will actually pull config from.

        #. If a runtime config file is specified, we will use that file.
        #. If no file is specified and there is a file named "hyrax_config.toml" in the cwd we will use it.
        #. If no file is specified and there is no file named "hyrax_config.toml" in the cwd we will
           exclusively work off the configuration defaults in the packaged "hyrax_default_config.toml" file.

        Parameters
        ----------
        runtime_config_filepath : Union[Path, str, None], optional
            Location of the supplied config file, by default None

        Returns
        -------
        Path
            Path to the configuration file ultimately used for config resolution. When we fall back to the
            package supplied default config file, the Path to that file is returned.

        Raises
        ------
        FileNotFoundError
            If a runtime config file is specified but does not exist.
        """
        if isinstance(runtime_config_filepath, str):
            runtime_config_filepath = Path(runtime_config_filepath)

        # If a runtime config file is explicitly specified, validate it exists
        if isinstance(runtime_config_filepath, Path) and not runtime_config_filepath.exists():
            raise FileNotFoundError(f"Cannot find config file {runtime_config_filepath}")

        # If a named config exists in cwd, and no config specified on cmdline, use cwd.
        if (
            runtime_config_filepath is None
            and DEFAULT_USER_CONFIG_FILEPATH.exists()
            and not ConfigManager._called_from_test
        ):
            runtime_config_filepath = DEFAULT_USER_CONFIG_FILEPATH

        if runtime_config_filepath is None:
            runtime_config_filepath = DEFAULT_CONFIG_FILEPATH

        return runtime_config_filepath


def create_results_dir(config: ConfigDict, postfix: str) -> Path:
    """Creates a results directory for this run.

    Postfix is the verb name of the run e.g. (infer, train, etc)

    The directory is created within the results dir (set with config results_dir)
    and follows the pattern <timestamp>-<postfix>

    The resulting directory is returned.

    Parameters
    ----------
    config : ConfigDict
        The full runtime configuration for this run
    postfix : str
        The verb name of the run.

    Returns
    -------
    Path
        The path created by this function
    """
    results_root = Path(config["general"]["results_dir"]).expanduser().resolve()
    # This date format is chosen specifically to create a lexical search order
    # which matches the date order.
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # Generate 4 random ascii characters to avoid collisions from multiple hyrax processes
    # started in a shared filesystem environment.
    random_str = base64.urlsafe_b64encode(random.randbytes(3)).decode("ascii")
    directory = results_root / f"{timestamp}-{postfix}-{random_str}"
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def find_most_recent_results_dir(config: ConfigDict, verb: str) -> Optional[Path]:
    """Find the most recent results directory corresponding to a particular verb
    This is a best effort search in the currently configured results root.

    If result directories are created within 1 second of one another this function
    will return one of the directories but it is undefined which one it will return.

    This function may return None indicating it could not find a directory matching
    the query verb
    """
    results_root = Path(config["general"]["results_dir"]).expanduser().resolve()

    max_timestamp = 0
    best_path = None

    for path in results_root.glob(f"*-{verb}-*"):
        if path.is_dir():
            regex = r"([0-9]{8})-([0-9]{6})-.*"
            m = re.match(regex, path.name)

            if m is None:
                continue

            timestamp = int(m[1] + m[2])
            if timestamp > max_timestamp:
                max_timestamp = timestamp
                best_path = path

    return best_path


def log_runtime_config(runtime_config: ConfigDict, output_path: Path, file_name: str = "runtime_config.toml"):
    """Log a runtime configuration.

    Parameters
    ----------
    runtime_config : ConfigDict
        A ConfigDict object containing runtime configuration values.
    output_path : str
        The path to put the config file
    file_name : str, Optional
        Optional name for the config file, defaults to "runtime_config.toml"
    """
    with open(output_path / file_name, "w") as f:
        f.write(tomlkit.dumps(runtime_config))
