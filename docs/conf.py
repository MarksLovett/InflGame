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
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives

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



sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../src"))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
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
html_css_files = [
    'css/math_fix.css',
]

html_sidebars = {
    "index": ["search-button-field"],
    "**": ["search-button-field", "sidebar-nav-bs"]
}

# MathJax: avoid automatic linebreaking against flex containers from the theme.
# (Equation crush after first paint is usually flex + CHTML measuring width=0.)
mathjax3_config = {
    "options": {
        "processHtmlClass": "tex2jax_process|mathjax_process|math|output_area",
    },
    "chtml": {
        "scale": 1,
        "displayAlign": "left",
        "displayIndent": "0",
    },
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


# -----------------------------------------------------------------------------
# Interactive Plotly embeds (Path B: HTML under _static + iframe)
# -----------------------------------------------------------------------------

_PLOTLY_STATIC_FILES = (
    "plotly/threed_gradient_ascent_paths_interactive.html",
)


class PlotlyIframe(Directive):
    """Embed a Plotly HTML asset from ``_static`` in an isolated iframe.

    Usage::

        .. plotly-iframe:: plotly/threed_gradient_ascent_paths_interactive.html
           :width: 100%
           :height: 560px
    """

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {
        "width": directives.unchanged,
        "height": directives.unchanged,
        "title": directives.unchanged,
    }

    def run(self):
        env = self.state.document.settings.env
        asset = self.arguments[0].lstrip("/")
        if asset.startswith("_static/"):
            asset = asset[len("_static/") :]
        # Page depth: api/adaptive/foo -> ../../_static/...
        depth = env.docname.count("/")
        rel = ("../" * depth) + "_static/" + asset
        width = self.options.get("width", "100%")
        height = self.options.get("height", "560px")
        title = self.options.get("title", "Interactive Plotly figure")
        html = (
            f'<div class="plotly-iframe-wrap" style="width:{width};max-width:100%;">'
            f'<iframe src="{rel}" title="{title}" '
            f'width="100%" height="{height}" '
            f'frameborder="0" loading="lazy" '
            f'style="border:0;border-radius:4px;"></iframe>'
            f"</div>"
        )
        return [nodes.raw("", html, format="html")]


def _ensure_plotly_static_assets(app):
    """Generate missing Plotly HTML assets before the HTML build copies _static."""
    if getattr(app.builder, "format", None) != "html":
        return
    static_root = Path(app.srcdir) / "_static"
    force = bool(os.environ.get("FORCE_PLOTLY_REGEN"))
    missing = [
        name for name in _PLOTLY_STATIC_FILES if not (static_root / name).exists()
    ]
    if not missing and not force:
        return
    script_dir = Path(app.srcdir).resolve().parent / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        import generate_plotly_docs as gen  # type: ignore

        dest = static_root / "plotly" / "threed_gradient_ascent_paths_interactive.html"
        gen.generate(out_path=dest)
    except Exception as exc:  # noqa: BLE001 — keep Sphinx build usable if generation fails
        app.warn(f"Failed to generate Plotly docs assets: {exc}")


def setup(app):
    app.add_directive("plotly-iframe", PlotlyIframe)
    app.connect("builder-inited", _ensure_plotly_static_assets)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
