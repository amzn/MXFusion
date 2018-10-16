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


def is_sampled_array(F, array):
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
    if is_sampled_array(F, array):
        return array
    else:
        return F.broadcast_axis(array, axis=0, size=num_samples)
