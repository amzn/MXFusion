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


import unittest
import json
from mxfusion.components import Variable
from mxfusion.util.serialization import ModelComponentDecoder, ModelComponentEncoder


class GraphSerializationTests(unittest.TestCase):
    """
    Tests the mxfusion.util.serialization classes for ModelComponent encoding and decoding.
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
