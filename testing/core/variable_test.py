import unittest
import mxnet as mx
import mxfusion.components as mfc
import mxfusion as mf


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
