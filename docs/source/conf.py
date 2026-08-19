# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import tomllib  # Python 3.11+
from pathlib import Path

# Add parent directory to path
docs_dir = Path(__file__).parent
project_root = docs_dir.parent.parent
sys.path.insert(0, str(project_root))

# Read metadata from pyproject.toml
pyproject_path = project_root / "pyproject.toml"
with pyproject_path.open("rb") as f:
    pyproject_data = tomllib.load(f)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = pyproject_data["project"]["name"]
author = pyproject_data["project"]["authors"][0]["name"]
release = version = pyproject_data["project"]["version"]
project_copyright = f"2024, {author}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx_copybutton",
    "myst_nb",  # MyST-NB includes MyST-parser functionality
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# HTML theme options for Furo
html_theme_options = {
    "source_repository": "https://github.com/finite-sample/rank-preserving-calibration",
    "source_branch": "main",
    "source_directory": "docs/source/",
}

# HTML context for additional customization
html_context = {
    "display_github": True,
    "github_user": "finite-sample",
    "github_repo": "rank_preserving_calibration",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
# Attributes: sections in the dataclass docstrings otherwise render as their
# own `.. attribute::` directives, which collide with the entries autodoc
# already emits for the same fields.
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings. `members` is deliberately absent: with it set here, a
# directive that names a restricted member list (api.md's calibration module)
# still pulled in every other member, and each class was then documented twice.
# `special-members` is likewise absent -- it made autodoc look for __init__ on
# a module.
autodoc_default_options = {
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# sphinx-copybutton configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_remove_prompts = True

# MyST-NB configuration
# The three example notebooks under examples/ do not run end to end: two
# reference variables that are never defined and one hits a genuine
# non-convergence. Executing them at build time therefore produced pages of
# tracebacks, and under `-W` it fails the build outright. Render them as
# authored until they are repaired.
nb_execution_mode = "off"
nb_execution_timeout = 300  # 5 minute timeout per cell
nb_execution_allow_errors = False  # Fail build on notebook errors
nb_execution_in_temp = False  # Execute in source directory

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# MyST-NB handles .ipynb files and MyST-parser handles .md files
# No need to explicitly set source_suffix as extensions handle this
