# -*- coding: utf-8 -*-
from datetime import datetime

import pkg_resources
import sys, os, shutil
from os.path import dirname, abspath

import recommonmark
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify

# copy over examples folder for notebook docs
EXAMPLES_SRC = "../examples"
EXAMPLES_DST = "examples"
if os.path.isdir(EXAMPLES_DST):
    shutil.rmtree(EXAMPLES_DST)
shutil.copytree(EXAMPLES_SRC, EXAMPLES_DST)
top_level = dirname(dirname(abspath(__file__)))
mxfusion_source = os.path.abspath('../mxfusion')
sys.path.insert(0, mxfusion_source)
sys.path.insert(1, top_level)


version = '1.0'
project = 'MXFusion'


# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest',
              'sphinx.ext.intersphinx', 'sphinx.ext.todo',
              'sphinx.ext.coverage', 'sphinx.ext.autosummary',
              'sphinx.ext.napoleon', 'nbsphinx', 'sphinx.ext.mathjax']
exclude_patterns = ['_build', '**.ipynb_checkpoints', '_templates', '**.cfg']

# Allow notebooks to have errors when generating docs
nbsphinx_allow_errors = True
nbsphinx_execute = 'never'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

source_suffix = ['.rst', '.md']  # The suffix of source filenames.
master_doc = 'index'  # The master toctree document.

source_parsers = {
   '.md': 'recommonmark.parser.CommonMarkParser',
}


copyright = '{} Amazon.com, Inc. or its affiliates. All Rights Reserved. SPDX-License-Identifier: \
             Apache-2.0'.format(datetime.now().year)

# The full version, including alpha/beta/rc tags.
release = version

# List of directories, relative to source directory, that shouldn't be searched
# for source files.
exclude_trees = ['_build']

pygments_style = 'sphinx'

autoclass_content = "class"
autodoc_default_flags = ['show-inheritance','members','undoc-members', 'classes']
autodoc_member_order = 'bysource'

# autosummary
autosummary_generate = True

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
htmlhelp_basename = '%sdoc' % project

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # this next line generates warnings about an unsupported option
    #'vcs_pageview_mode': '',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 8,
    'includehidden': True,
    'titles_only': True
}

html_logo = 'images/logos/blender-small.png'
html_favicon = 'images/logos/Favicon.png'

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {'http://docs.python.org/': None}

github_doc_root = ''
def setup(app):
    app.add_config_value('recommonmark_config', {
            'url_resolver': lambda url: github_doc_root + url,
            'enable_auto_toc_tree': True,
            }, True)
    app.add_transform(AutoStructify)
