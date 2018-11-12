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


import json
import mxfusion as mf
from ..common.exceptions import SerializationError


__GRAPH_JSON_VERSION__ = '1.0'


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
