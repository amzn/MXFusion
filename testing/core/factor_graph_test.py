import unittest
import uuid
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn
import mxfusion.components as mfc
from mxfusion.components.functions import MXFusionGluonFunction
import mxfusion as mf
from mxfusion.common.exceptions import ModelSpecificationError
from mxfusion.components.distributions.normal import Normal
from mxfusion.components import Variable
from mxfusion.models import Model
from mxfusion.components.variables.runtime_variable import add_sample_dimension, is_sampled_array, get_num_samples
from mxfusion.util.testutils import MockMXNetRandomGenerator


class FactorGraphTests(unittest.TestCase):
    """
    Tests the MXFusion.core.factor_graph.FactorGraph class.
    """

    def shape_match(self, old, new, var_map):
        if len(old) != len(new):
            return False
        for o, n in zip(old, new):
            if isinstance(n, mfc.Variable):
                if n != var_map[o]:
                    return False
            elif n != o:
                return False
        return True

    def make_model(self, net):
        component_set = set()
        m = mf.models.Model(verbose=False)
        m.N = mfc.Variable()
        m.f = MXFusionGluonFunction(net, num_outputs=1)
        m.x = mfc.Variable(shape=(m.N,))
        m.r = m.f(m.x)
        for k, v in m.r.factor.block_variables:
            if k.endswith('_weight') or k.endswith('_bias'):
                d = mf.components.distributions.Normal(mean=mx.nd.array([0]), variance=mx.nd.array([1e6]))
                v.set_prior(d)
                component_set.union(set([v,d]))
        m.y = mf.components.distributions.Categorical.define_variable(log_prob=m.r, num_classes=2, normalization=True, one_hot_encoding=False)
        component_set.union(set([m.N, m.f, m.x, m.r, m.y]))
        return m, component_set

    def make_simple_model(self):
        m = Model()
        mean = Variable()
        variance = Variable()
        m.r = Normal.define_variable(mean=mean, variance=variance)
        return m

    def make_net(self):
        D = 100
        net = nn.HybridSequential(prefix='hybrid0_')
        with net.name_scope():
            net.add(nn.Dense(D, activation="tanh"))
            net.add(nn.Dense(D, activation="tanh"))
            net.add(nn.Dense(2, flatten=True))
        net.initialize(mx.init.Xavier(magnitude=3))
        return net

    def setUp(self):
        self.TESTFILE = "testfile_" + str(uuid.uuid4()) + ".json"
        self.fg = mf.models.FactorGraph(name='test_fg')
        self.D = 10
        self.basic_net = nn.HybridSequential()
        with self.basic_net.name_scope():
            self.basic_net.add(nn.Dense(self.D, activation="relu"))
        self.bnn_net = self.make_net()

    def test_bnn_model(self):

        bnn_fg, component_set = self.make_model(self.bnn_net)
        self.assertTrue(component_set <= bnn_fg.components_graph.nodes().keys(),
                        "Variables are all added to _components_graph {} {}".format(component_set, bnn_fg.components_graph.nodes().keys()))
        self.assertTrue(component_set <= bnn_fg.components.keys(), "Variable is added to _components dict. {} {}".format(component_set, bnn_fg.components.keys()))
        # assert False

    def test_add_unresolved_components_distribution(self):
        v = mfc.Variable()
        m = mfc.Variable()
        f = Normal.define_variable(mean=m, variance=v)
        component_set = set((v, m, f))
        self.fg.f = f
        self.assertTrue(component_set <= self.fg.components_graph.nodes().keys(),
                        "Variables are all added to _components_graph {} {}".format(component_set, self.fg.components_graph.nodes().keys()))
        self.assertTrue(set((v.uuid, m.uuid, f.uuid)) <= self.fg.components.keys(), "Variable is added to _components dict. {} {}".format(v.uuid, self.fg.components))

    def test_add_unresolved_components_function(self):
        f = MXFusionGluonFunction(self.basic_net, num_outputs=1)
        x = mfc.Variable()
        y = f(x)
        component_set = set((x, y))
        self.fg.y = y
        self.assertTrue(component_set <= self.fg.components_graph.nodes().keys(),
                        "Variables are all added to _components_graph {} {}".format(component_set, self.fg.components_graph.nodes().keys()))
        self.assertTrue(set(map(lambda x: x.uuid, component_set)) <= self.fg.components.keys(), "Variable is added to _components dict. {} {}".format(set(map(lambda x: x.uuid, component_set)), self.fg.components.keys()))
        self.assertTrue(len(self.fg.components_graph.nodes().keys()) == 5, "There should be variables for the block parameters and a function evaluation totally 5 nodes in the graph but there were only {} in {}".format(len(self.fg.components_graph.nodes().keys()), self.fg.components_graph.nodes().keys()))

    def test_add_with_variable_success(self):
        v = mfc.Variable()
        self.fg.v = v
        self.assertTrue(v in self.fg.components_graph.nodes(), "Variable is added to _components_graph.")
        self.assertTrue(v.uuid in self.fg.components, "Variable is added to _components dict.")

    def test_assignment_with_variable_success(self):
        v = mfc.Variable()
        self.fg.v = v
        self.assertTrue(v in self.fg.components_graph.nodes(), "Variable is added to _components_graph.")
        self.assertTrue(v.uuid in self.fg.components, "Variable is added to _components dict.")

    def test_remove_variable_success(self):
        v = mfc.Variable()
        self.fg.v = v
        self.assertTrue(v in self.fg.components_graph.nodes(), "Variable is added to _components_graph.")
        self.assertTrue(v.uuid in self.fg.components, "Variable is added to _components dict.")
        self.fg.remove_component(v)
        self.assertFalse(v in self.fg.components_graph.nodes(), "Variable is removed from _components_graph.")
        self.assertFalse(v.uuid in self.fg.components, "Variable is removed from _components dict.")

    def test_remove_nonexistant_variable_failure(self):
        v = mfc.Variable()
        with self.assertRaises(ModelSpecificationError):
            self.fg.remove_component(v)

    def test_replicate_bnn_model(self):
        m, component_set = self.make_model(self.bnn_net)
        m2, var_map = m.clone()
        self.assertTrue(all([k.uuid == v.uuid for k, v in var_map.items()]))
        self.assertTrue(all([v in m.components for v in m2.components]), (set(m2.components) - set(m.components)))
        self.assertTrue(all([v in m2.components for v in m.components]), (set(m.components) - set(m2.components)))
        self.assertTrue(all([self.shape_match(m[i].shape, m2[i].shape, var_map) for i in m.variables]), (m.variables, m2.variables))

    def test_replicate_simple_model(self):
        m = mf.models.Model(verbose=False)
        m.x = mfc.Variable(shape=(2,))
        m.x_mean = mfc.Variable(value=mx.nd.array([0, 1]), shape=(2,))
        m.x_var = mfc.Variable(value=mx.nd.array([1e6]))
        d = mf.components.distributions.Normal(mean=m.x_mean, variance=m.x_var)
        m.x.set_prior(d)
        m2, var_map = m.clone()
        # compare m and m2 components and such for exactness.
        self.assertTrue(all([k.uuid == v.uuid for k, v in var_map.items()]))
        self.assertTrue(set([v.uuid for v in m.components.values()]) ==
                        set([v.uuid for v in m2.components.values()]))
        self.assertTrue(all([v in m.components for v in m2.components]), (set(m2.components) - set(m.components)))
        self.assertTrue(all([v in m2.components for v in m.components]), (set(m.components) - set(m2.components)))
        self.assertTrue(all([m.x.shape == m2.x.shape,
                             m.x_mean.shape == m2.x_mean.shape,
                             m.x_var.shape == m2.x_var.shape]))

    def test_set_prior_after_factor_attach(self):
        fg = mf.models.Model()
        d = mf.components.distributions.Normal(mean=mx.nd.array([1e2]), variance=mx.nd.array([1e6]))
        fg.d = d

        x = mfc.Variable()
        x.set_prior(d)
        self.assertTrue(set([v for _, v in d.successors]) == set([x]))
        self.assertTrue(set([v for _, v in x.predecessors]) == set([d]))
        self.assertTrue(x.graph == d.graph and d.graph == fg.components_graph)

    def test_compute_log_prob(self):
        m = Model()
        v = Variable(shape=(1,))
        m.v2 = Normal.define_variable(mean=v, variance=mx.nd.array([1]))
        m.v3 = Normal.define_variable(mean=m.v2, variance=mx.nd.array([1]), shape=(10,))
        np.random.seed(0)
        v_mx = mx.nd.array(np.random.randn(1))
        v2_mx = mx.nd.array(np.random.randn(1))
        v3_mx = mx.nd.array(np.random.randn(10))

        v_rt = add_sample_dimension(mx.nd, v_mx)
        v2_rt = add_sample_dimension(mx.nd, v2_mx)
        v3_rt = add_sample_dimension(mx.nd, v3_mx)
        variance = m.v2.factor.variance
        variance2 = m.v3.factor.variance
        variance_rt = add_sample_dimension(mx.nd, variance.constant)
        variance2_rt = add_sample_dimension(mx.nd, variance2.constant)
        log_pdf = m.compute_log_prob(F=mx.nd, variables={m.v2.uuid: v2_rt, m.v3.uuid:v3_rt, variance.uuid: variance_rt, variance2.uuid: variance2_rt, v.uuid: v_rt}).asscalar()

        variables = {m.v2.factor.mean.uuid: v_rt, m.v2.factor.variance.uuid: variance_rt, m.v2.factor.random_variable.uuid: v2_rt}
        log_pdf_1 = mx.nd.sum(m.v2.factor.log_pdf(F=mx.nd, variables=variables))
        variables = {m.v3.factor.mean.uuid: v2_rt, m.v3.factor.variance.uuid: variance2_rt, m.v3.factor.random_variable.uuid: v3_rt}
        log_pdf_2 = mx.nd.sum(m.v3.factor.log_pdf(F=mx.nd, variables=variables))

        assert log_pdf == (log_pdf_1 + log_pdf_2).asscalar()

    def test_draw_samples(self):
        np.random.seed(0)
        samples_1_np = np.random.randn(5)
        samples_1 = mx.nd.array(samples_1_np)
        samples_2_np = np.random.randn(50)
        samples_2 = mx.nd.array(samples_2_np)
        m = Model()
        v = Variable(shape=(1,))
        m.v2 = Normal.define_variable(mean=v, variance=mx.nd.array([1]), rand_gen=MockMXNetRandomGenerator(samples_1))
        m.v3 = Normal.define_variable(mean=m.v2, variance=mx.nd.array([0.1]), shape=(10,), rand_gen=MockMXNetRandomGenerator(samples_2))
        np.random.seed(0)
        v_np =np.random.rand(1)
        v_mx = mx.nd.array(v_np)

        v_rt = add_sample_dimension(mx.nd, v_mx)
        variance = m.v2.factor.variance
        variance2 = m.v3.factor.variance
        variance_rt = add_sample_dimension(mx.nd, variance.constant)
        variance2_rt = add_sample_dimension(mx.nd, variance2.constant)
        samples = m.draw_samples(F=mx.nd, num_samples=5, targets=[m.v3.uuid],
        variables={v.uuid: v_rt, variance.uuid: variance_rt, variance2.uuid: variance2_rt})[m.v3.uuid]

        samples_np = v_np + samples_1_np[:, None] + np.sqrt(0.1)*samples_2_np.reshape(5,10)

        assert is_sampled_array(mx.nd, samples) and get_num_samples(mx.nd, samples)==5
        assert np.allclose(samples.asnumpy(), samples_np)

    def test_reconcile_simple_model(self):
        m1 = self.make_simple_model()
        m2 = self.make_simple_model()
        component_map = mf.models.FactorGraph.reconcile_graphs([m1], m2)
        self.assertTrue(len(component_map) == len(m1.components))

    def test_reconcile_bnn_model(self):
        m1, _ = self.make_model(self.make_net())
        m2, _ = self.make_model(self.make_net())
        component_map = mf.models.FactorGraph.reconcile_graphs([m1], m2)
        self.assertTrue(len(component_map) == len(m1.components))

    def test_reconcile_model_and_posterior(self):
        x = np.random.rand(1000, 1)
        y = np.random.rand(1000, 1)
        x_nd, y_nd = mx.nd.array(y), mx.nd.array(x)

        net1 = self.make_net()
        net1(x_nd)
        net2 = self.make_net()
        net2(x_nd)
        m1, _ = self.make_model(net1)
        m2, _ = self.make_model(net2)

        from mxfusion.inference.meanfield import create_Gaussian_meanfield
        from mxfusion.inference import StochasticVariationalInference
        observed1 = [m1.y, m1.x]
        observed2 = [m2.y, m2.x]
        q1 = create_Gaussian_meanfield(model=m1, observed=observed1)
        alg1 = StochasticVariationalInference(num_samples=3, model=m1, posterior=q1, observed=observed1)
        q2 = create_Gaussian_meanfield(model=m2, observed=observed2)
        alg2 = StochasticVariationalInference(num_samples=3, model=m2, posterior=q2, observed=observed2)
        component_map = mf.models.FactorGraph.reconcile_graphs(
            alg1._graphs,
            primary_previous_graph=alg2._graphs[0],
            secondary_previous_graphs=[alg2._graphs[1]])
        assert len(set(component_map)) == \
            len(set(alg1.graphs[0].components.values()).union(
                set(alg1.graphs[1].components.values())))

    def test_save_reload_bnn_graph(self):
        m1, _ = self.make_model(self.make_net())
        m1.save(self.TESTFILE)
        m1_loaded = Model()
        m1_loaded.load_graph(self.TESTFILE)
        m1_loaded_edges = set(m1_loaded.components_graph.edges())
        m1_edges = set(m1.components_graph.edges())

        self.assertTrue(set(m1.components) == set(m1_loaded.components))
        self.assertTrue(set(m1.components_graph.edges()) == set(m1_loaded.components_graph.edges()), m1_edges.symmetric_difference(m1_loaded_edges))
        self.assertTrue(len(m1_loaded.components.values()) == len(set(m1_loaded.components.values())))
        import os
        os.remove(self.TESTFILE)

    def test_save_reload_then_reconcile_simple_graph(self):
        m1 = self.make_simple_model()
        m1.save(self.TESTFILE)
        m1_loaded = Model()
        m1_loaded.load_graph(self.TESTFILE)
        self.assertTrue(set(m1.components) == set(m1_loaded.components))

        m2 = self.make_simple_model()
        component_map = mf.models.FactorGraph.reconcile_graphs([m2], m1_loaded)
        self.assertTrue(len(component_map) == len(m1.components))
        sort_m1 = list(set(map(lambda x: x.uuid, m1.components.values())))
        sort_m1.sort()

        sort_m2 = list(set(map(lambda x: x.uuid, m2.components.values())))
        sort_m2.sort()

        sort_component_map_values = list(set(component_map.values()))
        sort_component_map_values.sort()

        sort_component_map_keys = list(set(component_map.keys()))
        sort_component_map_keys.sort()

        zippy_values = zip(sort_m2, sort_component_map_values)
        zippy_keys = zip(sort_m1, sort_component_map_keys)
        self.assertTrue(all([m1_item == component_map_item for m1_item, component_map_item in zippy_values]))
        self.assertTrue(all([m2_item == component_map_item for m2_item, component_map_item in zippy_keys]))

        import os
        os.remove(self.TESTFILE)

    def test_save_reload_then_reconcile_bnn_graph(self):
        m1, _ = self.make_model(self.make_net())
        m1.save(self.TESTFILE)
        m1_loaded = Model()
        m1_loaded.load_graph(self.TESTFILE)
        self.assertTrue(set(m1.components) == set(m1_loaded.components))

        m2, _ = self.make_model(self.make_net())
        component_map = mf.models.FactorGraph.reconcile_graphs([m2], m1_loaded)
        self.assertTrue(len(component_map.values()) == len(set(component_map.values())), "Assert there are only 1:1 mappings.")
        self.assertTrue(len(component_map) == len(m1.components))
        sort_m1 = list(set(map(lambda x: x.uuid, m1.components.values())))
        sort_m1.sort()

        sort_m2 = list(set(map(lambda x: x.uuid, m2.components.values())))
        sort_m2.sort()

        sort_component_map_values = list(set(component_map.values()))
        sort_component_map_values.sort()

        sort_component_map_keys = list(set(component_map.keys()))
        sort_component_map_keys.sort()

        zippy_values = zip(sort_m2, sort_component_map_values)
        zippy_keys = zip(sort_m1, sort_component_map_keys)
        self.assertTrue(all([m1_item == component_map_item for m1_item, component_map_item in zippy_values]))
        self.assertTrue(all([m2_item == component_map_item for m2_item, component_map_item in zippy_keys]))
        import os
        os.remove(self.TESTFILE)
