# Installation
## Dependencies / Prerequisites
MXFusion's primary dependencies are MXNet >= 1.2 and Networkx >= 2.1.
See [requirements](requirements/requirements.txt).

## Supported Architectures / Versions

MXFusion is tested on Python 3.5+ on MacOS and Amazon Linux.

## pip
If you just want to use MXFusion and not modify the source, you can install through pip:
```
pip install mxfusion
```

## From source
To install MXFusion from source, after cloning the repository run the following from the top-level directory:
```
pip install .
```

## Distributed Training
To allow distributed training of MXFusion using Horovod, install through pip (Note that MXFusion only support Horovod version below 0.18):
```
pip install horovod==0.16.4
```