# MXFusion
[![Build Status](https://travis-ci.org/amzn/MXFusion.svg?branch=master)](https://travis-ci.org/amzn/MXFusion) |
[![Build Status](https://travis-ci.org/amzn/MXFusion.svg?branch=develop)](https://travis-ci.org/amzn/MXFusion/branches) |
[![codecov](https://codecov.io/gh/amzn/MXFusion/branch/master/graph/badge.svg)](https://codecov.io/gh/amzn/MXFusion) |
[![pypi](https://img.shields.io/pypi/v/mxfusion.svg?style=flat)](https://pypi.org/project/mxfusion/) |
[![Documentation Status](https://readthedocs.org/projects/mxfusion/badge/?version=latest)](https://mxfusion.readthedocs.io/en/latest/?badge=latest) |
[![GitHub license](https://img.shields.io/github/license/amzn/mxfusion.svg)](https://github.com/amzn/mxfusion/blob/master/LICENSE)

![MXFusion](docs/images/logo/blender-small.png)

[Website](https://github.com/amzn/MXFusion) |
[Documentation](https://github.com/amzn/MXFusion/docs) |
[Contribution Guide](https://github.com/amzn/MXFusion/CONTRIBUTING.md)

MXFusion is a library for integrating probabilistic modelling with deep learning.

With MXFusion Modules you can use state-of-the-art inference techniques for specialized probabilistic models without needing to implement those techniques yourself. MXFusion helps you rapidly build and test new methods at scale, by focusing on the modularity of probabilistic models and their integration with modern deep learning techniques.

MXFusion uses  [MXNet](https://github.com/apache/incubator-mxnet) as its computational platform to bring the power of distributed, heterogenous computation to probabilistic modelling.

## Vision

TODO

### Why use probabilistic models?

TODO

## Features
It currently supports modelling of directed probabilistic models, deep learning integration through MXNet, and Variational Inference methods. Gaussian Processes are soon to come.


## Installation

### Dependencies / Prerequisites
MXFusion's primary dependencies are MXNet >= 1.2 and Networkx >= 2.1.
See [requirements](requirements/requirements.txt).

### Supported Architectures / Versions

MXFusion is tested on Python 3.5+ on MacOS and Amazon Linux.

### pip
If you just want to use MXFusion and not modify the source, you can install through pip:
```
pip install mxfusion
```

### From source
To install MXFusion from source, after cloning the repository run the following from the top-level directory:
```
pip install .
```

## Where to go from here?

[Documentation](https://github.com/amzn/MXFusion/docs)

[Contributions](CONTRIBUTING.md)

[Tutorials](Tutorials.md)

## Community
We welcome your contributions and questions and are working to build a responsive community around MXFusion. Feel free to file an Github issue if you find a bug or want to request a new feature.

## Contributing

Have a look at our [contributing guide](CONTRIBUTING.md), thanks for the interest!

Points of contact for MXFusion are:
* Eric Meissner (@meissnereric)
* Zhenwen Dai (@zhenwendai)

## License

MXFusion is licensed under Apache 2.0. See [LICENSE](LICENSE).
