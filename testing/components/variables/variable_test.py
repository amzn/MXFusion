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
import mxnet as mx
import numpy as np
import mxfusion.components as mfc
import mxfusion as mf
import mxfusion.common.exceptions as mf_exception


class VariableTests(unittest.TestCase):
    """
    Tests the MXFusion.core.variable.Variable class.
    """
    def test_set_prior(self):
        m = mf.models.Model(verbose=False)
        m.x = mfc.Variable()
        component_set = set([m.x])
        self.assertTrue(component_set <= set(m._components_graph.nodes().keys()),
                        "Variables are all added to components_graph {} {}".format(component_set, m.components_graph.nodes().keys()))

        d = mf.components.distributions.Normal(mean=mx.nd.array([0]), variance=mx.nd.array([1e6]))
        m.x.set_prior(d)
        component_set.union(set([d] + [v for _, v in d.inputs]))
        self.assertTrue(component_set <= set(m._components_graph.nodes().keys()),
                        "Variables are all added to components_graph {} {}".format(component_set, m.components_graph.nodes().keys()))

    def test_replicate_variable(self):
        m = mf.models.Model(verbose=False)
        m.x = mfc.Variable()
        def func(component):
            return 'recursive', 'recursive'
        x2 = m.x.replicate(replication_function=func)
        self.assertTrue(x2.uuid == m.x.uuid)

    def test_replicate_variable_with_variable_shape(self):
        m = mf.models.Model(verbose=False)
        y = mfc.Variable()
        m.x = mfc.Variable(shape=(y, 1))
        var_map = {}
        def func(component):
            return 'recursive', 'recursive'
        x2 = m.x.replicate(var_map=var_map, replication_function=func)
        self.assertTrue(x2.uuid == m.x.uuid)
        self.assertTrue(x2.shape == m.x.shape, (x2.shape, m.x.shape))
        self.assertTrue(y in m)

    def test_array_variable_shape(self):
        mxnet_array_shape = (3, 2)
        numpy_array_shape = (10, )
        mxnet_array = mx.nd.zeros(shape=mxnet_array_shape)
        numpy_array = np.zeros(shape=numpy_array_shape)

        # Test Case 1: Shape param not explicitly passed to Variable class
        variable = mf.Variable(value=mxnet_array)
        self.assertTrue(variable.shape == mxnet_array_shape)
        variable = mf.Variable(value=numpy_array)
        self.assertTrue(variable.shape == numpy_array_shape)

        # Test Case 2: Correct shape passed to Variable class
        variable = mf.Variable(value=mxnet_array, shape=mxnet_array_shape)
        self.assertTrue(variable.shape == mxnet_array_shape)
        variable = mf.Variable(value=numpy_array, shape=numpy_array_shape)
        self.assertTrue(variable.shape == numpy_array_shape)

        # Test Case 3: Incorrect shape passed to Variable class
        incorrect_shape = (1234, 1234)
        self.assertRaises(mf_exception.ModelSpecificationError, mf.Variable, value=mxnet_array, shape=incorrect_shape)
        self.assertRaises(mf_exception.ModelSpecificationError, mf.Variable, value=numpy_array, shape=incorrect_shape)
