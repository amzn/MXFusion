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

import numpy as np
from scipy.stats import norm
from ..module import Module
from ...models import Model, Posterior
from ...components.variables.variable import Variable
from ...components.variables.var_trans import PositiveTransformation
from ...components.distributions import GaussianProcess, ConditionalGaussianProcess, Bernoulli
from ...inference.variational import VariationalInference
from ...inference.forward_sampling import ForwardSamplingAlgorithm
from ...inference.inference_alg import SamplingAlgorithm
from ...util.customop import make_diagonal
from ...components.distributions.random_gen import MXNetRandomGenerator
from ...components.functions import MXFusionGluonFunction
import mxnet as mx
import mxnet.gluon.nn as nn


def compute_marginal_f(F, L, Kuf, Kff_diag, mu_u, Ls, mean_prior_u=None, mean_prior_f=None):
    # Compute q(f) parameters
    Kmmi = F.linalg.potri(L)
    KmmiKuf = F.linalg.gemm2(Kmmi, Kuf, alpha=1.0)

    if mean_prior_u is not None and mean_prior_f is not None:
        mu_f = mean_prior_f + F.linalg.gemm2(mu_u - mean_prior_u, KmmiKuf, alpha=1.0)
    else:
        mu_f = F.linalg.gemm2(mu_u, KmmiKuf, alpha=1.0)

    aux = Kff_diag - F.sum(KmmiKuf * Kuf, -2)

    KmmiKufbrod = mx.nd.expand_dims(KmmiKuf, axis=-3)
    KmmiKufbrod = mx.nd.broadcast_axis(KmmiKufbrod, axis=(KmmiKufbrod.ndim - 3), size=(Ls.shape[-3]))

    tmp = F.linalg.gemm2(Ls, KmmiKufbrod, transpose_a=True)

    v_f = mx.nd.expand_dims(aux, axis=-2) + F.sum(F.square(tmp), -2)

    return mu_f, v_f


def quadrature(F, f, dtype, mu_f, v_f, y=None, gh_points=None):
    if gh_points is None:
        gh_points = 20
    gh_x, gh_w = np.polynomial.hermite.hermgauss(gh_points)
    gh_w = gh_w / np.sqrt(np.pi)
    gh_x = mx.nd.array(gh_x, dtype=dtype)
    gh_w = mx.nd.array(gh_w, dtype=dtype)
    gh_x = mx.nd.expand_dims(gh_x, axis=0)
    gh_w = mx.nd.expand_dims(gh_w, axis=0)

    shape = mu_f.shape

    mu_f = mx.nd.reshape(mu_f, (-1, 1))
    v_f = mx.nd.reshape(v_f, (-1, 1))

    X = mx.nd.broadcast_mul(gh_x, mx.nd.sqrt(2. * v_f)) + mu_f

    if y is not None:
        y = tuple([mx.nd.reshape(y, (-1, 1))])
    else:
        y = tuple([])

    out_f = f(X, *y)
    out = F.linalg.gemm2(out_f, gh_w, transpose_b=True)

    return out.reshape(shape)


# TODO: Use Bernoulli logpdf plus link function
def dec_log_pdf(p, y):
    y_sign = 2 * y.flatten() - 1.0
    p = mx.nd.sigmoid(y_sign * p)
    p = mx.nd.clip(p, 1e-9, 1. - 1e-9)  # for numerical stability
    return mx.nd.log(p)


# TODO: Use the Bernoulli mean function plus link function
def dec_mean_lik(x):
    p = mx.nd.sigmoid(x)
    p = mx.nd.clip(p, 1e-9, 1. - 1e-9)  # for numerical stability
    return p


# TODO: Use link function
def get_confidence_alpha(mu_f, v_f, alpha=0.95):
    if alpha <= 0.5:
        raise ValueError("alpha should be higher than 0.5. Instead got alpha={}.".format(alpha))
    lb = mx.nd.sigmoid(norm.ppf(1 - alpha) * mx.nd.sqrt(v_f) + mu_f)
    ub = mx.nd.sigmoid(norm.ppf(alpha) * mx.nd.sqrt(v_f) + mu_f)
    return lb, ub

class SVGPClassificationLogPdf(VariationalInference):
    """
    The inference algorithm for computing the variational lower bound of the stochastic variational Gaussian process with Bernoulli likelihood.
    """
    def __init__(self, model, posterior, observed, lik_func, jitter=0.0):
        super(SVGPClassificationLogPdf, self).__init__(
            model=model, posterior=posterior, observed=observed)
        self.log_pdf_scaling = 1
        if jitter < 0.:
            raise ValueError("jitter should be >= 0. Instead got jitter = {}".format(jitter))
        self.jitter = jitter
        self.lik_func = lik_func

    def compute(self, F, variables):
        X = variables[self.model.X]
        Y = variables[self.model.Y]
        Z = variables[self.model.inducing_inputs]

        mu = variables[self.posterior.qU_mean]
        S_W = variables[self.posterior.qU_cov_W]
        S_diag = variables[self.posterior.qU_cov_diag]

        D = Y.shape[-1]
        M = Z.shape[-2]

        mean_prior_f = None
        mean_prior_u = None
        if self.model.mean_func is not None:
            mean_prior_f = self.model.mean_func(F, X)
            mean_prior_u = self.model.mean_func(F, Z)

        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter
        Kuf = kern.K(F, Z, X, **kern_params)
        Kff_diag = kern.Kdiag(F, X, **kern_params)
        L = F.linalg.potrf(Kuu)

        S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)
        Ls = F.linalg.potrf(S)

        mu_f, v_f = compute_marginal_f(F, L, Kuf, Kff_diag, mu, Ls, mean_prior_u=mean_prior_u, mean_prior_f = mean_prior_f)

        # Expectation with respect to the variational distribution q(f)
        logLik = quadrature(F, dec_log_pdf, Z.dtype, mu_f, v_f, Y.T).sum()

        # KL
        Lbrod = mx.nd.expand_dims(L, axis=1)
        Lbrod = mx.nd.broadcast_axis(Lbrod, axis=(Lbrod.ndim - 3), size=(Ls.shape[-3]))
        LinvLs = F.linalg.trsm(Lbrod, Ls, alpha=1.0)

        if self.model.mean_func is not None:
            mu = mu - mean_prior_u

        Linvmu = F.linalg.trsm(Lbrod, mx.nd.expand_dims(mu, axis=-1), alpha=1.0)

        KL_u = 0.5 * (
                F.sum(F.square(LinvLs)) +
                F.sum(F.square(Linvmu)) -
                D * M +
                D * 2 * F.linalg.sumlogdiag(L) -
                2 * F.linalg.sumlogdiag(Ls).sum()
        )

        # marginal posterior
        logL = logLik - self.model.U.factor.log_pdf_scaling * KL_u

        return logL


class SVGPClassificationMeanIntConfidencePrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, jitter=0.):
        super(SVGPClassificationMeanIntConfidencePrediction, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        if jitter < 0.:
            raise ValueError("jitter should be >= 0. Instead got jitter = {}".format(jitter))
        self.jitter = jitter

    def compute(self, F, variables):
        X = variables[self.model.X]
        Z = variables[self.model.inducing_inputs]

        mu = variables[self.graphs[1].qU_mean]
        S_W = variables[self.graphs[1].qU_cov_W]
        S_diag = variables[self.graphs[1].qU_cov_diag]

        M = Z.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter
        Kuf = kern.K(F, Z, X, **kern_params)
        Kff_diag = kern.Kdiag(F, X, **kern_params)
        L = F.linalg.potrf(Kuu)

        S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)
        Ls = F.linalg.potrf(S)

        mu_f, v_f = compute_marginal_f(F, L, Kuf, Kff_diag, mu, Ls)

        # Expectation with respect to the variational distribution q(f)
        mu = quadrature(F, dec_mean_lik, Z.dtype, mu_f, v_f)

        # Compute confidence intervals
        lb, ub = get_confidence_alpha(mu_f, v_f, alpha=0.95)

        outcomes = {self.model.Y.uuid: (mu, lb, ub)}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class SVGPClassificationSamplingPrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, num_samples=1, rand_gen=None, jitter=0.):
        super(SVGPClassificationSamplingPrediction, self).__init__(
            model=model, observed=observed, num_samples=num_samples, extra_graphs=[posterior])
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        if jitter < 0.:
            raise ValueError("jitter should be >= 0. Instead got jitter = {}".format(jitter))
        self.jitter = jitter

    def compute(self, F, variables):
        X = variables[self.model.X]
        Z = variables[self.model.inducing_inputs]

        mu = variables[self.graphs[1].qU_mean]
        S_W = variables[self.graphs[1].qU_cov_W]
        S_diag = variables[self.graphs[1].qU_cov_diag]

        M = Z.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter
        Kuf = kern.K(F, Z, X, **kern_params)
        Kff_diag = kern.Kdiag(F, X, **kern_params)
        L = F.linalg.potrf(Kuu)

        S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)
        Ls = F.linalg.potrf(S)

        mu_f, v_f = compute_marginal_f(F, L, Kuf, Kff_diag, mu, Ls)

        # Expectation with respect to the variational distribution q(f)
        mu = quadrature(F, dec_mean_lik, Z.dtype, mu_f, v_f)

        samples = self._rand_gen.sample_bernoulli(prob_true=mu, shape=(self.num_samples,) + mu.shape[1:],  dtype=self.model.F.factor.dtype)
        outcomes = {self.model.Y.uuid: samples}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class SVGPClassification(Module):
    """
    Stochastic variational sparse Gaussian process classification module

    This module contains a variational sparse Gaussian process with Bernoulli likelihood.

    :param X: the input variables on which the random variables are conditioned.
    :type X: Variable
    :param kernel: the kernel of Gaussian process.
    :type kernel: Kernel
    :param inducing_inputs: the inducing inputs of the sparse GP (optional). This variable will be auto-generated if not specified.
    :type inducing_inputs: Variable
    :param inducing_num: the number of inducing points of sparse GP (default: 10)
    :type inducing_num: int
    :param mean_func: the mean function of Gaussian process.
    :type mean_func: MXFusionFunction
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """

    def __init__(self, X, kernel, inducing_inputs=None,
                 inducing_num=10, mean_func=None,
                 rand_gen=None, dtype=None, ctx=None):

        if not isinstance(X, Variable):
            X = Variable(value=X)

        if inducing_inputs is None:
            inducing_inputs = Variable(shape=(inducing_num, kernel.input_dim), initial_value=Z_init)

        inputs = [('X', X), ('inducing_inputs', inducing_inputs)]

        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']

        super(SVGPClassification, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, dtype=dtype, ctx=ctx)

        self.mean_func = mean_func
        self.kernel = kernel
        self.lik_func = Bernoulli


    def _generate_outputs(self, output_shapes=None):
        """
        Generate the output of the module with given output_shapes.

        :param output_shape: the shapes of all the output variables
        :type output_shape: {str: tuple}
        """
        if output_shapes is None:
            Y_shape = self.X.shape[:-1] + (1,)
        else:
            Y_shape = output_shapes['random_variable']
        self.set_outputs([Variable(shape=Y_shape)])

    def _build_module_graphs(self):
        """
        Generate a model graph for GP classification module.
        """

        Y = self.random_variable
        graph = Model(name='sparsegp_classfication')
        graph.X = self.X.replicate_self()
        graph.inducing_inputs = self.inducing_inputs.replicate_self()
        M = self.inducing_inputs.shape[0]
        D = Y.shape[-1]

        graph.U = GaussianProcess.define_variable(
            X=graph.inducing_inputs, kernel=self.kernel,
            shape=(graph.inducing_inputs.shape[0], Y.shape[-1]),
            mean_func=self.mean_func, rand_gen=self._rand_gen, dtype=self.dtype,
            ctx=self.ctx)
        graph.F = ConditionalGaussianProcess.define_variable(
            X=graph.X, X_cond=graph.inducing_inputs, Y_cond=graph.U,
            kernel=self.kernel, shape=Y.shape, mean_func=self.mean_func,
            rand_gen=self._rand_gen, dtype=self.dtype, ctx=self.ctx)
        graph.Y = Y.replicate_self()

        sig = nn.HybridLambda(function='sigmoid')
        graph.link_function = MXFusionGluonFunction(sig, num_outputs=1, broadcastable=True)

        graph.F_trans = graph.link_function(graph.F)

        graph.Y.set_prior(
            Bernoulli(prob_true=graph.F_trans, rand_gen=self._rand_gen,
                   dtype=self.dtype, ctx=self.ctx)
        )

        graph.mean_func = self.mean_func
        graph.kernel = graph.U.factor.kernel

        post = Posterior(graph)
        post.qU_cov_diag = Variable(shape=(D, M), transformation=PositiveTransformation())
        post.qU_cov_W = Variable(shape=(D, M, M))
        post.qU_mean = Variable(shape=(D, M))
        return graph, [post]

    def _attach_default_inference_algorithms(self):
        """
        Attach the default inference algorithms for SVGPClassification Module:
        log_pdf <- SVGPClassificationLogPdf
        draw_samples <- ForwardSamplingAlgorithm
        prediction <- SVGPClassificationMeanIntConfidencePrediction
        """
        observed = [v for k, v in self.inputs] + \
            [v for k, v in self.outputs]
        self.attach_log_pdf_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SVGPClassificationLogPdf(
                self._module_graph, self._extra_graphs[0], observed, self.lik_func),
            alg_name='svgp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_draw_samples_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=ForwardSamplingAlgorithm(
                self._module_graph, observed),
            alg_name='svgp_sampling')

        self.attach_prediction_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SVGPClassificationMeanIntConfidencePrediction(
                self._module_graph, self._extra_graphs[0], observed),
            alg_name='svgp_predict')

    @staticmethod
    def define_variable(X, kernel, shape=None, inducing_inputs=None,
                        inducing_num=10, mean_func=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a variable drawn from a Stochastic variational sparse Gaussian process classification with Bernoulli likelihood.

        :param X: the input variables on which the random variables are conditioned.
        :type X: Variable
        :param kernel: the kernel of Gaussian process.
        :type kernel: Kernel
        :param shape: the shape of the random variable(s) (the default shape is
        the same shape as *X* but the last dimension is changed to one.)
        :type shape: tuple or [tuple]
        :param inducing_inputs: the inducing inputs of the sparse GP (optional). This variable will be auto-generated if not specified.
        :type inducing_inputs: Variable
        :param inducing_num: the number of inducing points of sparse GP (default: 10)
        :type inducing_num: int
        :param mean_func: the mean function of Gaussian process.
        :type mean_func: MXFusionFunction
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """
        gp = SVGPClassification(
            X=X, kernel=kernel,
            inducing_inputs=inducing_inputs, inducing_num=inducing_num,
            mean_func=mean_func, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        rep = super(SVGPClassification, self).replicate_self(attribute_map)

        rep.kernel = self.kernel.replicate_self(attribute_map)
        rep.mean_func = self.mean_func.replicate_self(attribute_map)

        return rep
