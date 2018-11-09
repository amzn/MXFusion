# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================


import mxnet

DEFAULT_DTYPE = 'float32'
MXNET_DEFAULT_MODE = mxnet.ndarray
MXNET_DEFAULT_DEVICE = None


def get_default_dtype():
    """
    Return the default dtype. The default dtype is float32.

    :returns: the default dtype
    :rtypes: str
    """
    return DEFAULT_DTYPE


def get_default_MXNet_mode():
    """
    Return the default MXNet mode

    :returns: an MXNet ndarray or symbol, indicating the default mode.
    :rtype: :class:`mxnet.symbol` or :class:`mxnet.ndarray`
    """
    return MXNET_DEFAULT_MODE


def get_default_device():
    """
    Return the default MXNet device

    :returns: an MXNet cpu or gpu, indicating the default device.
    """
    if MXNET_DEFAULT_DEVICE:
        return MXNET_DEFAULT_DEVICE
    else:
        return mxnet.context.Context.default_ctx
