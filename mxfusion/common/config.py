try:
    import configparser
except:
    import ConfigParser as configparser
import os
import mxnet

config = configparser.ConfigParser()
config.read(os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'config.cfg'))

def get_default_dtype():
    return config['defaults']['mxnet_dtype']


def get_default_MXNet_mode():
    """
    Return the default MXNet mode

    :returns: an MXNet ndarray or symbol, indicating the default mode.
    :rtype: :class:`mxnet.symbol` or :class:`mxnet.ndarray`
    """
    if config['defaults']['mxnet_mode'] == 'mxnet_ndarray':
        return mxnet.ndarray
    elif config['defaults']['mxnet_mode'] == 'mxnet_symbol':
        return mxnet.symbol


def get_default_device():
    """
    Return the default MXNet device

    :returns: an MXNet cpu or gpu, indicating the default device.
    """
    if config['defaults']['mxnet_device'] == 'cpu':
        return mxnet.cpu()
    elif config['defaults']['mxnet_device'] == 'gpu':
        return mxnet.gpu()
