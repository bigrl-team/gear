# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GEAR"
copyright = "2023, Hanjing Wang, Man-Kit Sit, Congjie He, Ying Wen, Weinan Zhang, Jun Wang, Yaodong Yang, Luo Mai"
author = "Hanjing Wang, Man-Kit Sit, Congjie He, Ying Wen, Weinan Zhang, Jun Wang, Yaodong Yang, Luo Mai"
release = "v0.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "recommonmark",
    "sphinx_markdown_tables",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_theme_options = {"light_css_variables": {}}
html_static_path = ["_static"]
html_theme_options = {
    "light_logo": "figs/icon.jpg",
    "dark_logo": "figs/icon.jpg",
    "source_repository": "https://github.com/bigrl-team/gear",
    "source_branch": "docs",
    "source_directory": "docs/",
}

intersphinx_mapping = {'python': ('https://docs.python.org/3', None)}

# autodoc
import importlib

importlib.import_module("torch")

autodoc_default_options = {"member-order": "groupwise"}
