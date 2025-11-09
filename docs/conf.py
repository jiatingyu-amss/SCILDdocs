import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "SCILD"
author = "Jiating Yu"
release = "1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",  
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# extensions.append("sphinx.ext.mathjax")

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


html_theme = "sphinx_rtd_theme"
html_logo = "images/SCILDlogo.png"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}
html_static_path = ["_static"]

