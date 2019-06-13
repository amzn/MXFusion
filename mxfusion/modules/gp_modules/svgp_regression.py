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
from ..module import Module
from ...models import Model, Posterior
from ...components.variables.variable import Variable
from ...components.variables.var_trans import PositiveTransformation
from ...components.distributions import GaussianProcess, Normal, ConditionalGaussianProcess
from ...inference.variational import VariationalInference
from ...inference.forward_sampling import ForwardSamplingAlgorithm
from ...inference.inference_alg import SamplingAlgorithm
from ...util.customop import make_diagonal
from ...util.customop import broadcast_to_w_samples
from ...components.distributions.random_gen import MXNetRandomGenerator
from ...components.variables.runtime_variable import broadcast_sample_dimension
from ...components.functions.operators import broadcast_to


class SVGPRegressionLogPdf(VariationalInference):
    """
    The inference algorithm for computing the variational lower bound of the stochastic variational Gaussian process
    with Gaussian likelihood.
    """
    def __init__(self, model, posterior, observed, jitter=0.):
        super(SVGPRegressionLogPdf, self).__init__(
            model=model, posterior=posterior, observed=observed)
        self.log_pdf_scaling = 1
        self.jitter = jitter

    def compute(self, F, variables):
        has_mean = self.model.F.factor.has_mean
        X = variables[self.model.X]
        Y = variables[self.model.Y]
        Z = variables[self.model.inducing_inputs]
        noise_var = variables[self.model.noise_var]
        mu = variables[self.posterior.qU_mean]
        S_W = variables[self.posterior.qU_cov_W]
        S_diag = variables[self.posterior.qU_cov_diag]

        D = Y.shape[-1]
        M = Z.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, Y, Z, noise_var, mu, S_W, S_diag, kern_params = broadcast_sample_dimension(
            [X, Y, Z, noise_var, mu, S_W, S_diag, kern_params])

        if noise_var.ndim == 2:  # it is heteroscedastic noise, when ndim == 3
            noise_var = F.expand_dims(noise_var, axis=-2)

        if noise_var.shape[-1] == 1:
            beta_sum = D*F.sum(1/noise_var, axis=-1)
        else:
            beta_sum = F.sum(1/noise_var, axis=-1)

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.expand_dims(F.eye(M, dtype=Z.dtype), axis=0) * \
                self.jitter
        Kuf = kern.K(F, Z, X, **kern_params)
        Kff_diag = kern.Kdiag(F, X, **kern_params)

        S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)

        if has_mean:
            mean = variables[self.model.mean]
            Y = Y - mean

        psi1Y = F.linalg.gemm2(Kuf, Y/noise_var, False, False)
        L = F.linalg.potrf(Kuu)
        Ls = F.linalg.potrf(S)
        LinvLs = F.linalg.trsm(L, Ls)
        Linvmu = F.linalg.trsm(L, mu)
        LinvKuf = F.linalg.trsm(L, Kuf)

        KfuKuuInvmu = F.linalg.gemm2(LinvKuf, Linvmu, True, False)
        KfuKuuInvLs = F.linalg.gemm2(LinvKuf, LinvLs, True, False)

        LinvKufY = F.linalg.trsm(L, psi1Y)

        KL_u = (M/2. + F.linalg.sumlogdiag(Ls))*D - F.linalg.sumlogdiag(L)*D\
            - F.sum(F.sum(F.square(LinvLs), axis=-1), axis=-1)/2.*D \
            - F.sum(F.sum(F.square(Linvmu), axis=-1), axis=-1)/2.

        logL = -F.sum(F.sum(F.square(Y)/noise_var + np.log(2. * np.pi) +
                            F.log(noise_var), axis=-1), axis=-1)/2.
        logL = logL - F.sum(Kff_diag*beta_sum, axis=-1)/2.
        logL = logL - F.sum(F.sum(F.square(KfuKuuInvmu)/noise_var, axis=-1),
                            axis=-1)/2.
        logL = logL - F.sum(F.sum(F.square(KfuKuuInvLs)*F.expand_dims(beta_sum, axis=-1), axis=-1),
                            axis=-1)/2.
        logL = logL + F.sum(F.sum(F.square(LinvKuf)*F.expand_dims(beta_sum, axis=-2), axis=-1),
                            axis=-1)/2.
        logL = logL + F.sum(F.sum(Linvmu*LinvKufY, axis=-1), axis=-1)
        logL = self.log_pdf_scaling*logL + KL_u
        return logL


class SVGPRegressionMeanVariancePrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, noise_free=True,
                 diagonal_variance=True, jitter=0.):
        super(SVGPRegressionMeanVariancePrediction, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        self.jitter = jitter
        self.noise_free = noise_free
        self.diagonal_variance = diagonal_variance

    def compute(self, F, variables):
        """
        The method for the computation of the sampling algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """
        has_mean = self.model.F.factor.has_mean
        X = variables[self.model.X]
        N = X.shape[-2]
        Z = variables[self.model.inducing_inputs]
        noise_var = variables[self.model.noise_var]
        mu = variables[self.graphs[1].qU_mean]
        S_W = variables[self.graphs[1].qU_cov_W]
        S_diag = variables[self.graphs[1].qU_cov_diag]
        M = Z.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter

        L = F.linalg.potrf(Kuu)
        Ls = F.linalg.potrf(S)
        LinvLs = F.linalg.trsm(L, Ls)
        Linvmu = F.linalg.trsm(L, mu)
        LinvSLinvT = F.linalg.syrk(LinvLs)
        wv = F.linalg.trsm(L, Linvmu, transpose=True)

        Kxt = kern.K(F, Z, X, **kern_params)
        mu = F.linalg.gemm2(Kxt, wv, True, False)
        if has_mean:
            mean = variables[self.model.mean]
            mu = mu + mean

        LinvKxt = F.linalg.trsm(L, Kxt)
        if self.diagonal_variance:
            Ktt = kern.Kdiag(F, X, **kern_params)
            tmp = F.linalg.gemm2(LinvSLinvT, LinvKxt)
            var = Ktt - F.sum(F.square(LinvKxt), axis=-2) + \
                F.sum(tmp*LinvKxt, axis=-2)
            var = F.expand_dims(var, axis=-1)
            if not self.noise_free:
                var = var + noise_var
        else:
            Ktt = kern.K(F, X, **kern_params)
            tmp = F.linalg.gemm2(LinvSLinvT, LinvKxt)
            var = Ktt - F.linalg.syrk(LinvKxt, True) + \
                F.linalg.gemm2(LinvKxt, tmp, True, False)
            var = F.expand_dims(var, axis=-1)
            if not self.noise_free:
                var = var + \
                    F.reshape(F.eye(N, dtype=X.dtype), shape=(1, N, N, 1)) * \
                    F.expand_dims(noise_var, axis=-2)

        outcomes = {self.model.Y.uuid: (mu, var)}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class SVGPRegressionSamplingPrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, rand_gen=None,
                 noise_free=True, diagonal_variance=True, jitter=0.):
        super(SVGPRegressionSamplingPrediction, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        self.noise_free = noise_free
        self.diagonal_variance = diagonal_variance
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        self.jitter = jitter

    def compute(self, F, variables):
        """
        The method for the computation of the sampling algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """
        has_mean = self.model.F.factor.has_mean
        X = variables[self.model.X]
        N = X.shape[-2]
        Z = variables[self.model.inducing_inputs]
        noise_var = variables[self.model.noise_var]
        mu = variables[self.graphs[1].qU_mean]
        S_W = variables[self.graphs[1].qU_cov_W]
        S_diag = variables[self.graphs[1].qU_cov_diag]
        M = Z.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter

        L = F.linalg.potrf(Kuu)
        Ls = F.linalg.potrf(S)
        LinvLs = F.linalg.trsm(L, Ls)
        Linvmu = F.linalg.trsm(L, mu)
        LinvSLinvT = F.linalg.syrk(LinvLs)
        wv = F.linalg.trsm(L, Linvmu, transpose=True)

        Kxt = kern.K(F, Z, X, **kern_params)
        mu = F.linalg.gemm2(Kxt, wv, True, False)
        if has_mean:
            mean = variables[self.model.mean]
            mu = mu + mean

        LinvKxt = F.linalg.trsm(L, Kxt)
        if self.diagonal_variance:
            Ktt = kern.Kdiag(F, X, **kern_params)
            tmp = F.linalg.gemm2(LinvSLinvT, LinvKxt)
            var = Ktt - F.sum(F.square(LinvKxt), axis=-2) + \
                F.sum(tmp*LinvKxt, axis=-2)
            if not self.noise_free:
                var += noise_var
            die = self._rand_gen.sample_normal(shape=(self.num_samples,) + mu.shape[1:],
                                               dtype=self.model.F.factor.dtype)
            samples = mu + die * F.sqrt(F.expand_dims(var, axis=-1))
        else:
            Ktt = kern.K(F, X, **kern_params)
            tmp = F.linalg.gemm2(LinvSLinvT, LinvKxt)
            cov = Ktt - F.linalg.syrk(LinvKxt, True) + \
                F.linalg.gemm2(LinvKxt, tmp, True, False)
            if not self.noise_free:
                cov += F.eye(N, dtype=X.dtype) * noise_var
            L = F.linalg.potrf(cov)
            out_shape = (self.num_samples,) + mu.shape[1:]
            L = broadcast_to_w_samples(F, L, out_shape[:-1] + out_shape[-2:-1])

            die = self._rand_gen.sample_normal(shape=out_shape,
                                               dtype=self.model.F.factor.dtype)
            samples = mu + F.linalg.trmm(L, die)

        outcomes = {self.model.Y.uuid: samples}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class SVGPRegression(Module):
    """
    Stochastic variational sparse Gaussian process regression module

    This module contains a variational sparse Gaussian process with Gaussian likelihood.

    :param X: the input variables on which the random variables are conditioned.
    :type X: Variable
    :param kernel: the kernel of Gaussian process.
    :type kernel: Kernel
    :param noise_var: the variance of the Gaussian likelihood
    :type noise_var: Variable
    :param inducing_inputs: the inducing inputs of the sparse GP (optional). This variable will be auto-generated
    if not specified.
    :type inducing_inputs: Variable
    :param num_inducing: the number of inducing points of sparse GP (default: 10)
    :type num_inducing: int
    :param mean: the mean of Gaussian process.
    :type mean: Variable
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """

    def __init__(self, X, kernel, noise_var, inducing_inputs=None,
                 num_inducing=10, mean=None,
                 rand_gen=None, dtype=None, ctx=None):
        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)
        if inducing_inputs is None:
            inducing_inputs = Variable(
                shape=(num_inducing, kernel.input_dim),
                initial_value=np.random.randn(num_inducing, kernel.input_dim))
        inputs = [('X', X), ('inducing_inputs', inducing_inputs),
                  ('noise_var', noise_var)]
        input_names = [k for k, _ in inputs]
        if mean is not None:
            inputs.append(('mean', mean))
            input_names.append('mean')
            self._has_mean = True
        else:
            self._has_mean = False
        output_names = ['random_variable']
        super(SVGPRegression, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, dtype=dtype, ctx=ctx)
        self.kernel = kernel

    def _generate_outputs(self, output_shapes=None):
        """
        Generate the output of the module with given output_shapes.

        :param output_shapes: the shapes of all the output variables
        :type output_shapes: {str: tuple}
        """
        if output_shapes['random_variable'] is None:
            Y_shape = self.X.shape[:-1] + (1,)
        else:
            Y_shape = output_shapes['random_variable']
        self.set_outputs([Variable(shape=Y_shape)])

    def _build_module_graphs(self):
        """
        Generate a model graph for GP regression module.
        """
        Y = self.random_variable
        graph = Model(name='sparsegp_regression')
        graph.X = self.X.replicate_self()
        graph.inducing_inputs = self.inducing_inputs.replicate_self()
        M = self.inducing_inputs.shape[0]
        graph.noise_var = self.noise_var.replicate_self()
        graph.U = GaussianProcess.define_variable(
            X=graph.inducing_inputs, kernel=self.kernel,
            shape=(graph.inducing_inputs.shape[0], Y.shape[-1]),
            rand_gen=self._rand_gen, dtype=self.dtype, ctx=self.ctx)
        if self._has_mean:
            mean = self.mean.replicate_self()
            graph.mean = mean
        else:
            mean = None
        graph.F = ConditionalGaussianProcess.define_variable(
            X=graph.X, X_cond=graph.inducing_inputs, Y_cond=graph.U,
            kernel=self.kernel, shape=Y.shape, mean=mean,
            rand_gen=self._rand_gen, dtype=self.dtype, ctx=self.ctx)
        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(
            mean=graph.F, variance=broadcast_to(graph.noise_var, graph.Y.shape)))
        graph.kernel = graph.U.factor.kernel
        post = Posterior(graph)
        post.qU_cov_diag = Variable(shape=(M,), transformation=PositiveTransformation())
        post.qU_cov_W = Variable(shape=(M, M))
        post.qU_mean = Variable(shape=(M, Y.shape[-1]))
        return graph, [post]

    def _attach_default_inference_algorithms(self):
        """
        Attach the default inference algorithms for SVGPRegression Module:
        log_pdf <- SVGPRegressionLogPdf
        draw_samples <- ForwardSamplingAlgorithm
        prediction <- SVGPRegressionMeanVariancePrediction
        """
        observed = [v for k, v in self.inputs] + \
            [v for k, v in self.outputs]
        self.attach_log_pdf_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SVGPRegressionLogPdf(
                self._module_graph, self._extra_graphs[0], observed),
            alg_name='svgp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_draw_samples_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=ForwardSamplingAlgorithm(
                self._module_graph, observed),
            alg_name='svgp_sampling')

        self.attach_prediction_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SVGPRegressionMeanVariancePrediction(
                self._module_graph, self._extra_graphs[0], observed),
            alg_name='svgp_predict')

    @staticmethod
    def define_variable(X, kernel, noise_var, shape=None, inducing_inputs=None,
                        num_inducing=10, mean=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a variable drawn from a Stochastic variational sparse Gaussian process regression with
        Gaussian likelihood.

        :param X: the input variables on which the random variables are conditioned.
        :type X: Variable
        :param kernel: the kernel of Gaussian process.
        :type kernel: Kernel
        :param noise_var: the variance of the Gaussian likelihood
        :type noise_var: Variable
        :param shape: the shape of the random variable(s) (the default shape is
        the same shape as *X* but the last dimension is changed to one.)
        :type shape: tuple or [tuple]
        :param inducing_inputs: the inducing inputs of the sparse GP (optional). This variable will be auto-generated
        if not specified.
        :type inducing_inputs: Variable
        :param num_inducing: the number of inducing points of sparse GP (default: 10)
        :type num_inducing: int
        :param mean: the mean of Gaussian process.
        :type mean: Variable
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """
        gp = SVGPRegression(
            X=X, kernel=kernel, noise_var=noise_var,
            inducing_inputs=inducing_inputs, num_inducing=num_inducing,
            mean=mean, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        rep = super(SVGPRegression, self).replicate_self(attribute_map)

        rep.kernel = self.kernel.replicate_self(attribute_map)
        rep._has_mean = self._has_mean
        return rep
