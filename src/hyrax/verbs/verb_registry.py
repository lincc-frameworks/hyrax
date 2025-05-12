import logging
from abc import ABC
from typing import Optional

logger = logging.getLogger(__name__)


class Verb(ABC):  # noqa: B024
    """Base class for all hyrax verbs"""

    # Verbs get to define how their parser gets added to the main parser
    # This is given in case verbs do not define any keyword args for
    # subparser.add_parser()
    add_parser_kwargs: dict[str, str] = {}

    def __init__(self, config):
        """
        .. py:method:: __init__

        Overall initialization for all verbs that saves the config
        """
        self.config = config


# Verbs with no class are assumed to have a function in hyrax.py which
# performs their function. All other verbs should be defined by named classes
# in hyrax.verbs and use the @hyrax_verb decorator
VERB_REGISTRY: dict[str, Optional[type[Verb]]] = {
    "train": None,
    "infer": None,
    "download": None,
    "prepare": None,
    "rebuild_manifest": None,
}


def hyrax_verb(cls: type[Verb]) -> type[Verb]:
    """Decorator to register a hyrax verb"""
    from hyrax.plugin_utils import update_registry

    update_registry(VERB_REGISTRY, cls.cli_name, cls)  # type: ignore[attr-defined]
    return cls


def all_verbs() -> list[str]:
    """Returns all verbs that are currently registered"""
    return [verb for verb in VERB_REGISTRY]


def all_class_verbs() -> list[str]:
    """Returns all verbs that are currently registered with a class-based implementation"""
    return [verb for verb in VERB_REGISTRY if VERB_REGISTRY.get(verb) is not None]


def is_verb_class(cli_name: str) -> bool:
    """Returns true if the verb has a class based implementation

    Parameters
    ----------
    cli_name : str
        The name of the verb on the command line interface

    Returns
    -------
    bool
        True if the verb has a class-based implementation
    """
    return cli_name in VERB_REGISTRY and VERB_REGISTRY.get(cli_name) is not None


def fetch_verb_class(cli_name: str) -> Optional[type[Verb]]:
    """Gives the class object for the named verb

    Parameters
    ----------
    cli_name : str
        The name of the verb on the command line interface


    Returns
    -------
    Optional[type[Verb]]
        The verb class or None if no such verb class exists.
    """
    return VERB_REGISTRY.get(cli_name)
