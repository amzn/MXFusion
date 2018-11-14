import pytest
import os
import mxnet as mx
import numpy as np
from mxfusion.models import Model
from mxfusion.modules.gp_modules import SVGPClassification
from mxfusion.components.distributions.gp.kernels import RBF
from mxfusion.components import Variable
from mxfusion.inference import Inference, MAP, ModulePredictionAlgorithm, TransferInference
from mxfusion.components.variables.var_trans import PositiveTransformation
from mxfusion.modules.gp_modules.svgp_classification import SVGPClassificationSamplingPrediction


import matplotlib
matplotlib.use('Agg')
import GPy


class TestSVGPClassificationModule(object):

    def gen_data(self):
        np.random.seed(0)

        N = 10
        M = 3
        Dx = 3
        Dy = 2

        X = np.random.rand(N, Dx)
        Y = np.random.randint(2, size=(N, Dy))
        Z = np.random.rand(M, Dx)

        qU_mean = np.random.rand(Dy, M)
        qU_cov_W = np.random.rand(Dy, M, M)
        qU_cov_diag = np.random.rand(Dy, M)

        lengthscale = np.random.rand(Dx)
        variance = np.random.rand(1)

        return Dy, X, Y, Z, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag

        # TODO: In order to compare to GPy, we need GPy to be able to use the sigmoid link function
        # qU_chol = [
        #     np.linalg.cholesky(qU_cov_W[dd, :, :].dot(qU_cov_W[dd, :, :].T) +
        #     np.diag(qU_cov_diag[dd, :]))[None, :, :] for dd in range(Dy)
        # ]
        # qU_chol = np.concatenate(qU_chol, axis=0)
        #
        # return Dy, X, Y, Z, lengthscale, variance, qU_mean, \
        #     qU_cov_W, qU_cov_diag, qU_chol

    def gen_mxfusion_model(self, dtype, Dy, Z, lengthscale, variance,
                           rand_gen=None):
        m = Model()
        m.N = Variable()
        m.X = Variable(shape=(m.N, 3))
        m.Z = Variable(shape=(3, 3), initial_value=mx.nd.array(Z, dtype=dtype))
        kernel = RBF(input_dim=3, ARD=True,
                     variance=mx.nd.array(variance, dtype=dtype),
                     lengthscale=mx.nd.array(lengthscale, dtype=dtype),
                     dtype=dtype)

        m.Y = SVGPClassification.define_variable(X=m.X, kernel=kernel,
                                                 inducing_inputs=m.Z,
                                                 shape=(m.N, Dy), dtype=dtype)
        gp = m.Y.factor
        return m, gp


    def test_log_pdf(self):
        Dy, X, Y, Z, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag = self.gen_data()

        dtype = 'float64'
        m, gp = self.gen_mxfusion_model(dtype, Dy, Z, lengthscale,
                                        variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        l_mf, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        assert np.allclose(l_mf.asnumpy(), 66.660689)

    # FIXME: In order to compare to GPy, we need GPy to be able to use the sigmoid link function
    #     m_gpy = GPy.core.SVGP(X=X, Y=Y, Z=Z,
    #                           kernel=GPy.kern.RBF(Dx, ARD=True, lengthscale=lengthscale, variance=variance),
    #                           likelihood=GPy.likelihoods.Bernoulli())
    #     m_gpy.q_u_mean = qU_mean.T
    #     m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)
    #     l_gpy = m_gpy.log_likelihood()
    #     assert np.allclose(l_mf.asnumpy(), l_gpy)

    def test_prediction(self):
        # TODO: It only checks consistency. Add unit tests to check the logic.
        np.random.seed(0)

        Dy, X, Y, Z, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag = self.gen_data()

        Nt = 5
        Xt = np.random.rand(Nt, X.shape[1])

        dtype = 'float64'
        m, gp = self.gen_mxfusion_model(dtype, Dy, Z, lengthscale,
                                        variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        infr2 = TransferInference(ModulePredictionAlgorithm(m, observed=[m.X], target_variables=[m.Y]),
                                  infr_params=infr.params, dtype=np.float64)
        res = infr2.run(X=mx.nd.array(Xt, dtype=dtype))[0]

        mu_mf, lb_mf, ub_mf = res[0].asnumpy()[0], res[1].asnumpy()[0], res[2].asnumpy()[0]

        pred_mf = np.concatenate((mu_mf, lb_mf, ub_mf), axis=0)
        pred_ser = np.genfromtxt(os.path.join("testing","modules", "svgp_classification_predictions.csv"), delimiter=',')[1:, 1:]

        assert np.allclose(pred_mf, pred_ser)

        # FIXME: In order to compare to GPy, we need GPy to be able to use the sigmoid link function
        # Xt = np.random.rand(Nt, X.shape[1])
        #
        # m_gpy = GPy.core.SVGP(X=X, Y=Y, Z=Z, kernel=GPy.kern.RBF(3, ARD=True, lengthscale=lengthscale, variance=variance), likelihood=GPy.likelihoods.Bernoulli())
        # m_gpy.q_u_mean = qU_mean.T
        # m_gpy.q_u_chol = GPy.util.choleskies.triang_to_flat(qU_chol)
        #
        # mu_gpy, _ = m_gpy.predict(Xt)
        # assert np.allclose(mu_gpy, mu_mf.T), (mu_gpy, mu_mf.T)

    def test_sampling_prediction(self):
        # TODO: It only checks consistency. Add unit tests to check the logic.
        np.random.seed(0)

        Dy, X, Y, Z, lengthscale, variance, qU_mean, \
            qU_cov_W, qU_cov_diag = self.gen_data()

        Nt = 5
        Xt = np.random.rand(Nt, X.shape[1])
        num_samples = 5

        dtype = 'float64'
        m, gp = self.gen_mxfusion_model(dtype, Dy, Z, lengthscale,
                                        variance)

        observed = [m.X, m.Y]
        infr = Inference(MAP(model=m, observed=observed), dtype=dtype)
        infr.initialize(X=X.shape, Y=Y.shape)
        infr.params[gp._extra_graphs[0].qU_mean] = mx.nd.array(qU_mean, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_W] = mx.nd.array(qU_cov_W, dtype=dtype)
        infr.params[gp._extra_graphs[0].qU_cov_diag] = mx.nd.array(qU_cov_diag, dtype=dtype)

        loss, _ = infr.run(X=mx.nd.array(X, dtype=dtype), Y=mx.nd.array(Y, dtype=dtype))

        infr_pred = TransferInference(ModulePredictionAlgorithm(model=m, observed=[m.X], target_variables=[m.Y],
                                                                num_samples=num_samples), infr_params=infr.params)
        gp = m.Y.factor
        gp.attach_prediction_algorithms(
            targets=gp.output_names, conditionals=gp.input_names,
            algorithm=SVGPClassificationSamplingPrediction(
                gp._module_graph, gp._extra_graphs[0], [gp._module_graph.X]),
            alg_name='svgp_predict')
        y_samples_pred = infr_pred.run(X=mx.nd.array(Xt, dtype=dtype))[0].asnumpy()
        y_samples_ser = np.genfromtxt(os.path.join("testing","modules", "svgp_classification_samples.csv"), delimiter=',')[1:, 1:]
        y_samples_ser = y_samples_ser.reshape((num_samples, Dy, Nt))

        np.allclose(y_samples_pred, y_samples_ser)
