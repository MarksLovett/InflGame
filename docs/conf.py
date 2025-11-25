# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import math
import os
from os.path import relpath, dirname
import re
import sys
import warnings
from docutils import nodes
from docutils.parsers.rst import Directive

from intersphinx_registry import get_intersphinx_mapping
import matplotlib
import matplotlib.pyplot as plt
from numpydoc.docscrape_sphinx import SphinxDocString
from sphinx.util import inspect

import scipy
from scipy._lib._util import _rng_html_rewrite
# Workaround for sphinx-doc/sphinx#6573
# ua._Function should not be treated as an attribute
import scipy._lib.uarray as ua
from scipy.stats._distn_infrastructure import rv_generic
from scipy.stats._multivariate import multi_rv_generic



autodoc_mock_imports = ["tree", "tkinter"]



sys.path.insert(0,os.path.abspath(".."))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    'myst_nb',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The main toctree document.
master_doc = 'index'


# General substitutions.
project = 'InflGame'
copyright = '2025, Mark Lovett'
author = 'Mark Lovett'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []
exclude_patterns = [  # glob-style
    "**.ipynb",
]

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False


# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'sphinx'

# Ensure all our internal links work
#nitpicky = True
#nitpick_ignore = [
    ## This ignores errors for classes (OptimizeResults, sparse.dok_matrix)
    
    ## which inherit methods from `dict`. missing references to builtins get
    ## ignored by default (see https://github.com/sphinx-doc/sphinx/pull/7254),
    ## but that fix doesn't work for inherited methods.
    #("py:class", "a shallow copy of D"),
    #("py:class", "a set-like object providing a view on D's keys"),
    #("py:class", "a set-like object providing a view on D's items"),
    #("py:class", "an object providing a view on D's values"),
    #("py:class", "None.  Remove all items from D."),
    #("py:class", "(k, v), remove and return some (key, value) pair as a"),
    #("py:class", "None.  Update D from dict/iterable E and F."),
    #("py:class", "None.  Update D from mapping/iterable E and F."),
    #("py:class", "v, remove specified key and return the corresponding value."),
#]



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

html_theme = 'pydata_sphinx_theme'

html_sidebars = {
    "index": ["search-button-field"],
    "**": ["search-button-field", "sidebar-nav-bs"]
}




# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True



# -----------------------------------------------------------------------------
# Autodoc
# -----------------------------------------------------------------------------

autodoc_default_options = {
    'inherited-members': None,
}
autodoc_typehints = 'none'

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}
