import logging
import sys
from pathlib import Path
from typing import Optional, Union


class Hyrax:
    """
    Overall class that represents an interface into hyrax. Currently this encapsulates a configuration and is
    the external interface to all verbs in a programmatic or notebook context.

    CLI functions in hyrax_cli are implemented by calling this class
    """

    def __init__(self, *, config_file: Optional[Union[Path, str]] = None, setup_logging: bool = True):
        """
        .. py:method:: __init__

        Initialize hyrax. Always applies the default config, and merges it with any provided config file.

        Parameters
        ----------
        config_file : Union[Path, str], optional
            filename or pathlib.Path to a config file, by default None

        setup_logging : bool, optional
            Logging setup for a hyrax object is global loggers named "hyrax.*" If you want to turn off
            logging config for "hyrax.*" python loggers, pass False here. By default True.

        Notes
        -----
        You may want to set ``setup_logging`` manually if:
        - You have multiple Hyrax objects in your application or notebook, and would like to control
        which of their logging configs is used globally. By creating one of your objects with
        setup_logging=True and the others with setup_logging=False, the single object created with
        setup_logging=True will control where the log is emitted to and what the threshold level is.
        - You have another library which needs overall control over python logging's config, and you
        do not want hyrax to alter any global logging config. In this case you should always pass
        setup_logging=False. Hyrax will still send logs into python logging; however, the other
        system will be responsible for where those logs are emitted, and what the threshold level
        is.

        You may want to leave the default of setup_logging=True if:
        - You have a single Hyrax object in use at any time. This is true in most notebook like
        environments.

        """
        from .config_utils import ConfigManager

        self.config_manager = ConfigManager(runtime_config_filepath=config_file)
        self.config = self.config_manager.config

        # Configure our logger. We do not use __name__ here because that would give us a "hyrax.hyrax" logger
        # which would not aggregate logs from hyrax.downloadCutout which creates its own
        # "hyrax.downloadCutout" logger. It uses this name to preserve backwards compatibility with its CLI.
        #
        # Choosing "hyrax" as our log name also means that modules like hyrax.models, or hyrax.dataloaders
        # can get a logger with logging.getLogger(__name__) and will automatically roll up to us.
        #
        self.logger = logging.getLogger("hyrax")

        # The logger object we get from logging.getLogger() is a global process-level object.
        #
        # This creates some tension around what hyrax initialization ought happen when the log handlers for
        # hyrax have already been configured by another object!
        #
        # There are two imagined situations at time of writing where this tension occurs:
        #
        # 1) A notebook user has a cell with `hyrax_instance = hyrax.Hyrax()` and they run this multiple times
        #    in the same kernel process. If this person is changing any logging in their hyrax config file
        #    and rerunning (perhaps to debug an issue) they likely expect the most recently __init__()ed
        #    hyrax object to control their logging.
        #
        # 2) An application (or notebook) user configures multiple Hyrax instance objects intending to use
        #    several different configs. If the logging configs differ between those objects we don't have a
        #    clear choice about what the global hyrax logging config ought be, unless the user tells us
        #    somehow.
        #
        # In the face of this tension we allow a kwarg to be passed (setup_logging) which defaults to True
        #
        # In the default case setup_logging=True, User 1 gets their expected behavior where the most recently
        # initialized object controls global logging.
        #
        # User 2 can initialize some (or all!) of their objects with setup_logging = False to achieve
        # appropriate logging behavior for their application.
        #
        if setup_logging:
            # Remove all prior handlers
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)

            # Setup our handlers from config
            self._initialize_log_handlers()

        self.logger.debug(f"Runtime Config read from: {ConfigManager.resolve_runtime_config(config_file)}")

    def set_config(self, key: str, value):
        """
        Set a config value at runtime. This modifies the in-memory config object.
        Once the configuration is updated, the entire config is re-rendered to
        ensure that any requested external library default configs are incorporated.

        Would be used like:
        hyrax.set_config("model.name", "external_hyrax_example.example_model.ExampleModel")
        hyrax.set_config("train.epochs", 42)

        Parameters
        ----------
        key : str
            The dotted key to set, e.g. "model.name"

        value : Any
            The value to set the key to.
        """
        self.config_manager.set_config(key, value)  # type: ignore[arg-type]
        self.config = self.config_manager.config

    def _initialize_log_handlers(self):
        """Private initialization helper, Adds handlers and level setting to the global self.logger object"""

        general_config = self.config["general"]
        # default to warning level
        level = general_config["log_level"]
        if not isinstance(level, str):
            level = logging.WARNING
        else:
            # In python 3.11 and above you can do this:
            #
            # level_map = logging.getLevelNamesMapping()
            #
            # but for now we do this:
            level_map = {
                "critical": logging.CRITICAL,
                "error": logging.ERROR,
                "warning": logging.WARNING,
                "warn": logging.WARNING,
                "info": logging.INFO,
                "debug": logging.DEBUG,
            }
            level = level_map.get(level.lower(), logging.WARNING)

        self.logger.setLevel(level)

        # Default to stderr for destination
        log_destination = general_config["log_destination"]

        if not isinstance(log_destination, str):
            # Default to stderr in cases where configured log_destination has a non string object
            handler = logging.StreamHandler(stream=sys.stderr)
        else:
            if log_destination.lower() == "stdout":
                handler = logging.StreamHandler(stream=sys.stdout)
            elif log_destination.lower() == "stderr":
                handler = logging.StreamHandler(stream=sys.stderr)
            else:
                handler = logging.FileHandler(log_destination)

        self.logger.addHandler(handler)

        # Format our log messages
        formatter = logging.Formatter("[%(asctime)s %(name)s:%(levelname)s] %(message)s")
        handler.setFormatter(formatter)

    def raw_data_dimensions(self) -> tuple[list[int], list[int]]:
        """Gives the dimensions of underlying data that forms input to the training, and inference
        steps. This is the raw data that the data loader must normalize to the model

        Returns
        -------
        tuple[list[int],list[int]]
            widths and heights of all images available locally.
        """
        from .download import Downloader

        downloader = Downloader(config=self.config)
        manifest = downloader.get_manifest()
        widths = [int(dim[0]) for dim in manifest["dim"]]
        heights = [int(dim[1]) for dim in manifest["dim"]]
        return widths, heights

    def download(self, **kwargs):
        """
        See Hyrax.download.run()
        """
        from .download import Downloader

        downloader = Downloader(config=self.config)
        return downloader.run(**kwargs)

    def prepare(self, **kwargs):
        """
        See Hyrax.prepare.run()
        """
        from .prepare import run

        return run(config=self.config, **kwargs)

    def rebuild_manifest(self, **kwargs):
        """
        See Hyrax.rebuild_manifest.run()
        """
        from .rebuild_manifest import run

        return run(config=self.config, **kwargs)

    # Python notebook interface to class verbs
    # we need both __dir__ and __getattr__ so that the
    # functions from the various verb classes appear to be
    # methods on the hyrax object
    def __dir__(self):
        from .verbs.verb_registry import all_class_verbs

        return sorted(dir(Hyrax) + list(self.__dict__.keys()) + all_class_verbs())

    def __getattr__(self, name):
        from .verbs.verb_registry import fetch_verb_class, is_verb_class

        if not is_verb_class(name):
            return None

        # We return the run function on the verb class after
        # just-in-time creating the verb so that a notebook user
        # sees the function signature and help.
        #
        # It may be possible to do this with functools.partial techniques
        # but should be tested.
        verb_inst = fetch_verb_class(name)(config=self.config)
        return verb_inst.run
