import json
from ..components import ModelComponent
from ..common.exceptions import SerializationError


__GRAPH_JSON_VERSION__ = '1.0'


class ModelComponentEncoder(json.JSONEncoder):

    def default(self, obj):
        """
        Serializes a ModelComponent object. Note: does not serialize the successor attribute as it isn't  necessary for serialization.
        """
        import mxfusion.components as mf
        if isinstance(obj, mf.ModelComponent):
            return {
                "type": obj.__class__.__name__,
                "uuid": obj.uuid,
                "name": obj.name,
                "inherited_name": obj.inherited_name if hasattr(obj, 'inherited_name') else None,
                "attributes": [a.uuid for a in obj.attributes],
                "version": __GRAPH_JSON_VERSION__
            }
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
        v = ModelComponent()
        v._uuid = obj['uuid']
        v.attributes = obj['attributes']
        v.name = obj['name']
        v.inherited_name = obj['inherited_name']
        v.type = obj['type']
        return v
