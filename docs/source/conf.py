# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))



project = 'DeepTrees🌳'
copyright = '2025, Taimur Khan, Caroline Arnold, Harsh Grover'
author = 'Taimur Khan, Caroline Arnold, Harsh Grover'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
#html_theme='pydata_sphinx_theme'
html_static_path = ['_static']

extensions = [
    'sphinx.ext.autodoc',         # Auto-generate documentation from docstrings
    'sphinx.ext.autosummary',      # Create summary tables
    'sphinx.ext.napoleon',         # Support for Google and NumPy style docstrings
    'sphinx.ext.viewcode',         # Add links to source code
    'sphinx_autodoc_typehints',    # Show type hints in the docs (optional)
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": "__init__",
    "show-inheritance": True,
}

# Automatically generate summary files
autosummary_generate = True

# Napoleon settings (if using Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
