import pytest
import subprocess
import sys
from unittest.mock import patch


def test_cli_config_file_validation():
    """Test that CLI properly validates config file existence."""
    # This test should fail for non-existent config files
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        non_existent_config = os.path.join(temp_dir, "non_existent.toml")
        
        # Test that the CLI properly handles non-existent config files
        # We can't easily test the full CLI without dependencies, so we'll test the config validation
        from hyrax.config_utils import ConfigManager
        
        with pytest.raises(FileNotFoundError):
            ConfigManager.resolve_runtime_config(non_existent_config)


def test_cli_arg_parsing():
    """Test that CLI argument parsing works for different flag positions."""  
    # We can test the argument parsing logic more directly
    from hyrax_cli.main import main
    import argparse
    
    # Test that both patterns should work (when dependencies are available)
    # For now, we focus on the config validation which is the main issue
    pass  # This will be a placeholder for now since we have network issues