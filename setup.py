from setuptools import setup, find_packages
import re

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('mxfusion/__version__.py', 'r') as rv:
    text = rv.read().split('=')
    __version__ = re.search(r'\d+\.\d+\.\d+', text[-1]).group()

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
    install_requires=['networkx>=2.1', 'numpy>=1.7'],
    license='Apache License 2.0',
    classifiers=(
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
