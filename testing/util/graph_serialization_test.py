import unittest
import json
from mxfusion.components import Variable
from mxfusion.util.graph_serialization import ModelComponentDecoder, ModelComponentEncoder


class GraphSerializationTests(unittest.TestCase):
    """
    Tests the mxfusion.util.graph_serialization classes for ModelComponent encoding and decoding.
    """

    def test_encode_component(self):

        v = Variable()
        v2 = Variable()
        v.predecessors = [('first', v2)]
        # TODO: extend attributes to support generic representation
        v.attributes = [Variable()]#[('shape', (Variable(), 1))]

        data = {
            "var": v
        }

        a = json.dumps(data, cls=ModelComponentEncoder, indent=2)
        print(a)

    def test_decode_component(self):
        v = Variable()
        v2 = Variable()
        v.predecessors = [('first', v2)]
        v.attributes = [Variable()] #[('shape', (Variable(), 1))]

        data = {
            "var": v
        }
        a = json.dumps(data, cls=ModelComponentEncoder, indent=2)
        b = json.loads(a, cls=ModelComponentDecoder)
        assert b == data
