# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
from importlib.metadata import version

# Define path to the code to be documented **relative to where conf.py (this file) is kept**
sys.path.insert(0, os.path.abspath("../src/"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "hyrax"
copyright = "2024, LINCC Frameworks"
author = "LINCC Frameworks"
release = version("hyrax")
# for example take major/minor
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.mathjax", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]

extensions.append("autoapi.extension")
extensions.append("nbsphinx")
extensions.append("sphinx_tabs.tabs")

# -- sphinx-copybutton configuration ----------------------------------------
extensions.append("sphinx_copybutton")
## sets up the expected prompt text from console blocks, and excludes it from
## the text that goes into the clipboard.
copybutton_exclude = ".linenos, .gp"
copybutton_prompt_text = ">> "

## lets us suppress the copy button on select code blocks.
copybutton_selector = "div:not(.no-copybutton) > div.highlight > pre"

extensions.append("sphinx_togglebutton")

templates_path: list[str] = []
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# This assumes that sphinx-build is called from the root directory
master_doc = "index"
# Remove 'view source code' from top of page (for html, not python)
html_show_sourcelink = False
# Remove namespaces from class/method signatures
add_module_names = False

autoapi_type = "python"
autoapi_dirs = ["../src/hyrax", "../src/hyrax_cli"]
autoapi_ignore = ["*/__main__.py", "*/_version.py", "*3d_viz*"]  # , "*downloadCutout*"]
autoapi_add_toc_tree_entry = False
autoapi_member_order = "bysource"
# Useful for tracking down sphinx errors in autodoc's generated files from a sphinx warning
autoapi_keep_files = True
autoapi_python_class_content = "both"  # Render __init__ and class docstring concatenated.
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]


nitpick_ignore_regex = [
    # Packages that have their own docs
    (r"^py:.*", r"^abc\..*"),
    (r"^py:.*", r"^astropy\..*"),
    (r"^py:.*", r"^tomlkit\..*"),
    (r"^py:.*", r"^pathlib\..*"),
    (r"^py:.*", r"^Path.*"),
    (r"^py:.*", r"^Table.*"),
    (r"^py:.*", r"^torch\..*"),
    (r"^py:.*", r"^concurrent\..*"),
    (r"^py:.*", r"^numpy\..*"),
    (r"^py:.*", r"^npt\..*"),
    (r"^py:.*", r"^np\..*"),
    (r"^py:.*", r"^datetime\..*"),
    (r"^py:.*", r"^urllib\..*"),
    (r"^py:.*", r"^torchvision\..*"),
    (r"^py:.*", r"^collections\..*"),
    (r"^py:.*", r"^_collections_abc\..*"),
    (r"^py:.*", r"^nn\..*"),
    (r"^py:.*", r"^tensorboardX\..*"),
    (r"^py:.*", r"^ignite\..*"),
    (r"^py:.*", r"^pytorch-ignite\..*"),
    (r"^py:.*", r"^argparse\..*"),
    (r"^py:.*", r"^holoviews\..*"),
    (r"^py:.*", r"^hv\..*"),
    (r"^py:.*", r"^pd\..*"),
    (r"^py:.*", r"^threading\..*"),
    (r"^py:.*", r"^enum\..*"),
    # Types and idiomatic ways we document types
    (r"^py:.*", r"^T$"),
    (r"^py:class", r"^[oO]ptional[:]?$"),
    (r"^py:class", r"^tuple$"),
    (r"^py:class", r"^string$"),
    (r"^py:.*", r"^TOML.*"),
    (r"^py:.*", r"^Ellipsis.*"),
    (r"^py:.*", r"^ML [fF]ramework [mM]odel"),
    (r"^py:.*", r"^Tensor"),
    (r"^py:.*", r"^SummaryWriter"),
    (r"^py:.*", r"^Dataset"),
    (r"^py:.*", r"^Engine"),
    (r"^py:.*", r"^DataLoader"),
    (r"^py:.*", r"^DistributedDataParallel"),
    (r"^py:.*", r"^DataParallel"),
    (r"^py:.*", r"^ArgumentParser"),
    (r"^py:.*", r"^Namespace"),
    # Types defined by our package that autodoc misidentifies in annotations
    (r"^py:.*", r"^hyrax.data_sets.fits_image_dataset.files_dict$"),
    (r"^py:.*", r"^dim_dict$"),
    (r"^py:.*", r"^dC.Rect$"),
    (r"^py:.*", r"^hyrax.downloadCutout.downloadCutout.Rect$"),
]

html_theme = "sphinx_rtd_theme"
