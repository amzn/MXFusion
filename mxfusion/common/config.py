import mxnet

MXNET_DEFAULT_DTYPE = 'float32'
MXNET_DEFAULT_MODE = mxnet.ndarray
MXNET_DEFAULT_DEVICE = mxnet.cpu()

def get_default_dtype():
    return MXNET_DEFAULT_DTYPE


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
    return MXNET_DEFAULT_DEVICE
