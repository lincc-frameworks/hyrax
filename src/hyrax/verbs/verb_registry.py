import logging
from abc import ABC
from collections.abc import Mapping

logger = logging.getLogger(__name__)


class Verb(ABC):  # noqa: B024
    """Base class for all hyrax verbs"""

    # Verbs get to define how their parser gets added to the main parser
    # This is given in case verbs do not define any keyword args for
    # subparser.add_parser()
    add_parser_kwargs: dict[str, str] = {}

    # Subclasses declare which data_request groups they require or optionally use.
    # REQUIRED_DATA_GROUPS must all be present in the data_request config.
    # OPTIONAL_DATA_GROUPS are used when present but cause no error if absent.
    # Verbs that leave both empty skip data_request validation entirely.
    REQUIRED_DATA_GROUPS: tuple[str, ...] = ()
    OPTIONAL_DATA_GROUPS: tuple[str, ...] = ()
    cli_name = "VERB"
    description = ""

    def __init__(self, config):
        """
        .. py:method:: __init__

        Overall initialization for all verbs that saves the config
        """
        self.config = config
        self.validate_data_request()

    @classmethod
    def information(cls):
        """Returns a string describing this verb. Includes the following:
        - Name of the verb
        - Required Data Groups
        - Optional Data Groups
        - One line description of what this verb does

        If a data group is empty then it will be printed as an empty tuple.

        Returns
        -------
        str
            <name>: Data Groups: Req. (<req1>, <req2>, ...), Opt. (<opt1>, <opt2>, ...). <Description>
        """
        info = cls.cli_name + ": "
        required = "Req. " + str(cls.REQUIRED_DATA_GROUPS)
        optional = "Opt. " + str(cls.OPTIONAL_DATA_GROUPS)
        info = info + "Data groups: " + required + ", " + optional + ". " + cls.description
        return info

    def validate_data_request(self) -> None:
        """Validate the data_request configuration for this verb's known groups.

        Reads ``data_request`` (or the deprecated ``model_inputs``) from the
        verb's config and checks:

        1. All groups listed in ``REQUIRED_DATA_GROUPS`` are present.
        2. Cross-group split_fraction constraints (sum ≤ 1.0, consistency) hold
           for the active groups only — groups outside
           ``REQUIRED_DATA_GROUPS + OPTIONAL_DATA_GROUPS`` are ignored so that
           unrelated groups in a shared config do not cause false failures.

        Verbs that define neither ``REQUIRED_DATA_GROUPS`` nor
        ``OPTIONAL_DATA_GROUPS`` skip validation entirely.

        Raises
        ------
        RuntimeError
            If a required group is absent, or if cross-group split_fraction
            constraints are violated for the active groups.
        """
        if not self.REQUIRED_DATA_GROUPS and not self.OPTIONAL_DATA_GROUPS:
            return

        data_request = self.config.get("data_request") or self.config.get("model_inputs")
        if not data_request:
            return

        if not isinstance(data_request, Mapping):
            raise RuntimeError(
                f"{type(self).__name__} received a non-mapping data_request configuration "
                f"of type {type(data_request)!r}; expected a mapping from group name to config."
            )

        # Verify that every required group is present in the config.
        missing = [g for g in self.REQUIRED_DATA_GROUPS if g not in data_request]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__} requires dataset group(s) {missing} in the "
                f"data_request configuration, but they were not found. "
                f"Available groups: {sorted(data_request.keys())}."
            )

        # Build a DataRequestDefinition so we can call validate_cross_group.
        # If the stored config is structurally invalid, surface the problem as a
        # runtime error so that verb-time validation does not get silently skipped.
        from pydantic import ValidationError

        from hyrax.config_schemas.data_request import DataRequestDefinition

        try:
            definition = DataRequestDefinition.model_validate(data_request)
        except ValidationError as exc:
            raise RuntimeError(
                f"Invalid data_request configuration for {type(self).__name__}: {exc}"
            ) from exc

        # Restrict cross-group validation to the groups this verb actually uses.
        # Groups outside REQUIRED + OPTIONAL (e.g. 'infer' for a Train verb) are
        # ignored so that their configs cannot cause false validation failures.
        all_verb_groups = set(self.REQUIRED_DATA_GROUPS + self.OPTIONAL_DATA_GROUPS)
        active_groups = all_verb_groups & set(data_request.keys())
        try:
            definition.validate_cross_group(active_groups)
        except ValueError as exc:
            raise RuntimeError(f"Data request validation failed for {type(self).__name__}: {exc}") from exc


# Verbs with no class are assumed to have a function in hyrax.py which
# performs their function. All other verbs should be defined by named classes
# in hyrax.verbs and use the @hyrax_verb decorator
VERB_REGISTRY: dict[str, type[Verb] | None] = {
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


def fetch_verb_class(cli_name: str) -> type[Verb] | None:
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
