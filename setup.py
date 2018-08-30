import os
from setuptools import setup, find_packages
from mxfusion.__version__ import __version__


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='MXFusion',
      version=__version__,
      packages=find_packages(),
      package_data={'': ['config.cfg']}
      )
