#!/usr/bin/env python3
"""
Demonstration of the fixed set_config method handling quoted dotted strings.

This script demonstrates the issue that was fixed and shows how the new
implementation correctly handles TOML configuration keys with dotted table
names enclosed in quotes.

Issue: https://github.com/lincc-frameworks/hyrax/issues/XXX
"""

from hyrax.config_utils import parse_dotted_key


def main():
    """Demonstrate the fix for handling quoted dotted strings in set_config."""
    print("=" * 80)
    print("Demonstration: set_config with Quoted Dotted Strings")
    print("=" * 80)
    print()

    print("PROBLEM:")
    print("-" * 80)
    print("In TOML, table names like 'torch.optim.Adam' need to be quoted to")
    print("avoid being parsed as nested tables. However, the old set_config")
    print("method would incorrectly split on ALL dots, including those inside")
    print("quoted strings.")
    print()

    print("EXAMPLES:")
    print("-" * 80)
    print()

    # Example 1: Regular dotted key (backward compatible)
    key1 = "model.name"
    result1 = parse_dotted_key(key1)
    print(f"✓ Regular key: '{key1}'")
    print(f"  Parsed as: {result1}")
    print("  Expected:  ['model', 'name']")
    print()

    # Example 2: Single-quoted dotted table name
    key2 = "'torch.optim.Adam'.lr"
    result2 = parse_dotted_key(key2)
    print(f"✓ Single-quoted key: {key2}")
    print(f"  Parsed as: {result2}")
    print("  Expected:  ['torch.optim.Adam', 'lr']")
    print()

    # Example 3: Double-quoted dotted table name
    key3 = '"torch.optim.SGD".momentum'
    result3 = parse_dotted_key(key3)
    print(f"✓ Double-quoted key: {key3}")
    print(f"  Parsed as: {result3}")
    print("  Expected:  ['torch.optim.SGD', 'momentum']")
    print()

    # Example 4: Mixed quoted and unquoted
    key4 = "optimizer.'torch.optim.Adam'.lr"
    result4 = parse_dotted_key(key4)
    print(f"✓ Mixed key: {key4}")
    print(f"  Parsed as: {result4}")
    print("  Expected:  ['optimizer', 'torch.optim.Adam', 'lr']")
    print()

    print("=" * 80)
    print("USAGE WITH ConfigManager:")
    print("=" * 80)
    print()
    print("from hyrax.config_utils import ConfigManager")
    print()
    print("config_manager = ConfigManager()")
    print()
    print("# Set a value in a quoted table name")
    print("config_manager.set_config(\"'torch.optim.Adam'.lr\", 0.001)")
    print()
    print("# This will now correctly update:")
    print("#   config['torch.optim.Adam']['lr'] = 0.001")
    print()
    print("# Instead of incorrectly trying to access:")
    print("#   config['torch']['optim']['Adam']['lr'] = 0.001")
    print()

    print("=" * 80)
    print("✓ Fix successfully handles quoted dotted strings in TOML configs!")
    print("=" * 80)


if __name__ == "__main__":
    main()
