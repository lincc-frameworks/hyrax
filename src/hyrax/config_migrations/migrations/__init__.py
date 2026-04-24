import os
import pkgutil

# Automatically find all modules in the current directory. In the config_migrations
# __init__.py, we'll import ALL of the modules to trigger their @migration_step
# registration.
__all__ = [name for _, name, _ in pkgutil.iter_modules([os.path.dirname(__file__)])]
