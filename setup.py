import os
from setuptools import setup, find_packages
from mxfusion.__version__ import __version__

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements/requirements.txt', 'r') as req:
    requires = req.read().split("\n")

setup(
    name='MXFusion',  # this is the name of the package as you will import it i.e import package-name
    version=__version__,
    author='Eric Meissner',
    author_email='meissner.eric.7@gmail.com',
    description=' Modular Probabilistic Programming on MXNet',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/amzn/MXFusion',
    packages=find_packages(exclude=['testing*']),
    include_package_data=True,
    install_requires=requires,
    license='Apache License 2.0',
    classifiers=(
       # https://pypi.org/pypi?%3Aaction=list_classifiers
       'Development Status :: 5 - Production/Stable',
       'Intended Audience :: Developers',
       'Intended Audience :: Education',
       'Intended Audience :: Science/Research',
       'Programming Language :: Python :: 3',
       'Programming Language :: Python :: 3.4',
       'Programming Language :: Python :: 3.6',
       'Programming Language :: Python :: 3.7',
       'License :: OSI Approved :: Apache Software License',
       'Operating System :: OS Independent',
       'Topic :: Scientific/Engineering :: Artificial Intelligence',
       'Topic :: Scientific/Engineering :: Mathematics',
       'Topic :: Software Development :: Libraries',
       'Topic :: Software Development :: Libraries :: Python Modules'
    ),
)
