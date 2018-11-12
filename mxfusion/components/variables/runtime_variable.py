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


from mxnet.ndarray.ndarray import NDArray
from mxnet.symbol.symbol import Symbol


def add_sample_dimension(F, array):
    """
    Add an extra dimension with shape one in the front (axis 0) of an array representing samples.

    :param F: the execution mode of MXNet.
    :type F: mxnet.ndarray or mxnet.symbol
    :param array: the array that the extra dimension is added to.
    :type array: MXNet NDArray or MXNet Symbol
    :returns: the array with the extra dimension.
    :rtypes: the same type as the input array
    """
    return F.expand_dims(array, axis=0)


def add_sample_dimension_to_arrays(F, arrays, out=None):
    """
    Add the sample dimension to a dict of arrays.

    :param F: the execution mode of MXNet.
    :type F: mxnet.ndarray or mxnet.symbol
    :param arrays: a dictionary of MXNet arrays.
    :type arrays: {UUID: array}
    :param out: (optional) if not None, add processed arrays into out.
    :type out: dict
    """
    processed_arrays = {
        uuid: add_sample_dimension(F, v) if isinstance(v, (NDArray, Symbol))
        else v for uuid, v in arrays.items()}
    if out is not None:
        out.update(processed_arrays)
    return processed_arrays


def expectation(F, array):
    """
    Return the expectation across the samples if the variable is a set of samples, otherwise return the variable.

    :param F: the MXNet execution mode.
    :type F: mxnet.ndarray or mxnet.symbol
    """
    return F.mean(array, axis=0)


def array_has_samples(F, array):
    """
    Check if the array is a set of samples.

    :returns: True if the array is a set of samples.
    :rtypes: boolean
    """
    # TODO: replace array.shape with F.shape_array
    return array.shape[0] > 1


def get_num_samples(F, array):
    """
    Get the number of samples in the provided array. If the array is not a set of samples, the return value will be one.

    :returns: the number of samples.
    :rtypes: int
    """
    # TODO: replace array.shape with F.shape_array
    return array.shape[0]


def as_samples(F, array, num_samples):
    """
    Broadcast the variable as if it is a sampled variable. If the variable is already a sampled variable, it directly returns the data reference.

    :param F: the execution mode of MXNet.
    :type F: mxnet.ndarray or mxnet.symbol
    :param array: the array to operate on.
    :type array: MXNet NDArray or MXNet Symbol
    :param num_samples: the number of samples
    :type num_samples: int
    """
    if array_has_samples(F, array):
        return array
    else:
        return F.broadcast_axis(array, axis=0, size=num_samples)


def arrays_as_samples(F, arrays):
    """
    Broadcast the dimension of samples for a list of variables. If the number of samples of at least one of the variables is larger than one, all the variables in the list are broadcasted to have the same number of samples.

    :param F: the execution mode of MXNet.
    :type F: mxnet.ndarray or mxnet.symbol
    :param arrays: a list of arrays with samples to be broadcasted.
    :type arrays: [MXNet NDArray or MXNet Symbol or {str: MXNet NDArray or MXNet Symbol}]
    :returns: the list of variables after broadcasting
    :rtypes: [MXNet NDArray or MXNet Symbol or {str: MXNet NDArray or MXNet Symbol}]
    """
    num_samples = [max([get_num_samples(F, v) for v in a.values()]) if isinstance(a, dict) else get_num_samples(F, a) for a in arrays]
    max_num_samples = max(num_samples)
    if max_num_samples > 1:
        return [{k: as_samples(F, v, max_num_samples) for k, v in a.items()} if isinstance(a, dict) else as_samples(F, a, max_num_samples) for a in arrays]
    else:
        return arrays
