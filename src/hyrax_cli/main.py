import argparse
import sys
from importlib.metadata import version

from hyrax import Hyrax
from hyrax.verbs import all_verbs, fetch_verb_class, is_verb_class


def main():
    """Primary entry point for the Hyrax CLI. This handles dispatching to the various
    Hyrax actions and returning a result.
    """

    description = "Hyrax CLI"
    epilog = "Hyrax is the Framework for Image-Based Anomaly Detection"

    parser = argparse.ArgumentParser(description=description, epilog=epilog)
    _add_major_arguments(parser)

    # cut off "usage: " from beginning and "\n" from end so we get an invocation
    # which subcommand parsers can add to appropriately.
    subparser_usage_prefix = parser.format_usage()[7:-1]
    subparsers = parser.add_subparsers(title="Verbs:", required=False)

    # Add a subparser for every verb, (whether defined by function or class)
    for cli_name in all_verbs():
        subparser_kwargs = {}

        if is_verb_class(cli_name):
            verb_class = fetch_verb_class(cli_name)
            subparser_kwargs = verb_class.add_parser_kwargs

        verb_parser = subparsers.add_parser(
            cli_name, prog=subparser_usage_prefix + " " + cli_name, **subparser_kwargs
        )

        if is_verb_class(cli_name):
            verb_class.setup_parser(verb_parser)

        verb_parser.set_defaults(verb=cli_name)
        _add_major_arguments(verb_parser)

    args = parser.parse_args()

    if args.version:
        print(version("hyrax"))
        return

    if not args.verb:
        parser.print_help()
        sys.exit(1)

    hyrax_instance = Hyrax(config_file=args.runtime_config)
    retval = 0
    if is_verb_class(args.verb):
        verb = fetch_verb_class(args.verb)(hyrax_instance.config)
        retval = verb.run_cli(args)
    else:
        getattr(hyrax_instance, args.verb)()

    exit(retval)


def _add_major_arguments(parser):
    parser.add_argument("--version", dest="version", action="store_true", help="Show version")
    parser.add_argument("-c", "--runtime-config", type=str, help="Full path to runtime config file")


if __name__ == "__main__":
    main()
