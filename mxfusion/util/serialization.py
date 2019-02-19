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


import io
import json
import mxfusion as mf
import mxnet as mx
import numpy as np
import zipfile
from ..common.exceptions import SerializationError
from ..common.config import get_default_device


__GRAPH_JSON_VERSION__ = '1.0'
SERIALIZATION_VERSION = '2.0'
DEFAULT_ZIP = 'inference.zip'
FILENAMES = {
    'graphs' : 'graphs.json',
    'mxnet_params' : 'mxnet_parameters.npz',
    'mxnet_constants' : 'mxnet_constants.npz',
    'variable_constants' : 'variable_constants.json',
    'configuration' : 'configuration.json',
    'version_file' : 'version.json'
}
ENCODINGS = {
    'json' : 'json',
    'numpy' : 'numpy'
}

class ModelComponentEncoder(json.JSONEncoder):

    def default(self, obj):
        """
        Serializes a ModelComponent object. Note: does not serialize the successor attribute as it isn't  necessary for serialization.
        """
        if isinstance(obj, mf.components.ModelComponent):
            object_dict = obj.as_json()
            object_dict["version"] = __GRAPH_JSON_VERSION__
            object_dict["type"] = obj.__class__.__name__
            return object_dict
        return super(ModelComponentEncoder, self).default(obj)


class ModelComponentDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        """
        Reloads a ModelComponent object. Note: does not reload the successor attribute as it isn't necessary for serialization.
        """
        if not isinstance(obj, type({})) or 'uuid' not in obj:
            return obj
        if obj['version'] != __GRAPH_JSON_VERSION__:
            raise SerializationError('The format of the stored model component '+str(obj['name'])+' is from an old version '+str(obj['version'])+'. The current version is '+__GRAPH_JSON_VERSION__+'. Backward compatibility is not supported yet.')
        if 'graphs' in obj:
            v = mf.modules.Module(None, None, None, None)
            v.load_module(obj)
        else:
            v = mf.components.ModelComponent()
            v.inherited_name = obj['inherited_name'] if 'inherited_name' in obj else None
        v.name = obj['name']
        v._uuid = obj['uuid']
        v.attributes = obj['attributes']
        v.type = obj['type']
        return v

def load_json_file(target_file, decoder=None):
    with open(target_file) as f:
        return json.load(f, cls=decoder)

def load_json_from_zip(zip_filename, target_file, decoder=None):
    """
    Utility function that loads a json file from inside a zip file without unzipping the zip file
    and returns the loaded json as a dictionary.
    :param encoder: optional. a JSONDecoder class to pass to the json.load function for loading back in the dict.
    """
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        json_file = zip_file.open(target_file)
        # json.load only takes str in 3.4/3.5 so we read, decode to UTF-8, and convert to a StringIO
        loaded = json.load(io.StringIO(json_file.read().decode()), cls=decoder)
    return loaded

def make_numpy(obj):
    """
    Utility function that takes a dictionary of numpy or MXNet arrays and
    returns a dictionary of numpy arrays. Used to standardize serialization.
    """
    ERR_MSG = "This function shouldn't be called on anything except " + \
             " dictionaries of numpy and MXNet arrays."
    if not isinstance(obj, type({})):
        raise SerializationError(ERR_MSG)

    np_obj = {}
    for k,v in obj.items():
        if isinstance(v, np.ndarray):
            np_obj[k] = v
        elif isinstance(v, mx.ndarray.ndarray.NDArray):
            np_obj[k] = v.asnumpy()
        else:
            raise SerializationError(ERR_MSG)
    return np_obj

def load_parameters(npz_filename, zip_file, context=None):
    """
    Helper function to load the parameters from a npz file directly into a dictionary as mxnet arrays.
    """
    context = context if context is not None else get_default_device()
    params_file = zip_file.read(npz_filename)
    try:
        loaded = np.load(io.BytesIO(params_file))
    except OSError as e:
        """
        Numpy load doesn't handle reloading an empty .npz directory after savez so just continue with an empty
        dict if it throws an OSError here when loading back.
        See https://github.com/chainer/chainer/issues/4542
        """
        return {}
    parameters = {}
    for k,v in loaded.items():
        parameters[k] = mx.nd.array(v, dtype=v.dtype, ctx=context)
    return parameters
