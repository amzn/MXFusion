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
from mxnet import autograd
from ..module import Module
from ...models import Model, Posterior
from ...components.variables.variable import Variable
from ...components.distributions import GaussianProcess, Normal, \
    ConditionalGaussianProcess
from ...inference.forward_sampling import ForwardSamplingAlgorithm
from ...inference.variational import VariationalInference
from ...inference.inference_alg import SamplingAlgorithm
from ...util.customop import broadcast_to_w_samples
from ...components.distributions.random_gen import MXNetRandomGenerator
from ...components.variables.runtime_variable import arrays_as_samples


class SparseGPRegressionLogPdf(VariationalInference):
    """
    The inference algorithm for computing the variational lower bound of variational sparse Gaussian process.
    """
    def __init__(self, model, posterior, observed, jitter=0.):
        super(SparseGPRegressionLogPdf, self).__init__(
            model=model, posterior=posterior, observed=observed)
        self.jitter = jitter

    def compute(self, F, variables):
        X = variables[self.model.X]
        Y = variables[self.model.Y]
        Z = variables[self.model.inducing_inputs]
        noise_var = variables[self.model.noise_var]
        D = Y.shape[-1]
        M = Z.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, Y, Z, noise_var, kern_params = arrays_as_samples(
            F, [X, Y, Z, noise_var, kern_params])

        noise_var_m = F.expand_dims(noise_var, axis=-2)

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.expand_dims(F.eye(M, dtype=Z.dtype), axis=0) * \
                self.jitter

        Kuf = kern.K(F, Z, X, **kern_params)
        Kff_diag = kern.Kdiag(F, X, **kern_params)

        L = F.linalg.potrf(Kuu)
        LinvKuf = F.linalg.trsm(L, Kuf)

        A = F.expand_dims(F.eye(M, dtype=Z.dtype), axis=0) + \
            F.broadcast_div(F.linalg.syrk(LinvKuf), noise_var_m)
        LA = F.linalg.potrf(A)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            Y = Y - mean
        LAInvLinvKufY = F.linalg.trsm(LA, F.linalg.gemm2(LinvKuf, Y))

        logL = - D*F.linalg.sumlogdiag(LA)
        logL = logL - F.sum(F.sum(F.square(Y)/noise_var_m + np.log(2. * np.pi)
                                  + F.log(noise_var_m), axis=-1), axis=-1)/2
        logL = logL + F.sum(F.sum(
            F.square(LAInvLinvKufY)/(2*F.square(noise_var_m)), axis=-1),
            axis=-1)
        logL = logL - D*F.sum(Kff_diag/(2*noise_var), axis=-1)
        logL = logL + D*F.sum(F.sum(F.square(LinvKuf)/(2.*noise_var_m),
                              axis=-1), axis=-1)

        with autograd.pause():
            wv = F.broadcast_div(
                F.linalg.trsm(L, F.linalg.trsm(LA, LAInvLinvKufY,
                                               transpose=True),
                              transpose=True), noise_var_m)
            self.set_parameter(variables, self.graphs[1].wv, wv[0])
            self.set_parameter(variables, self.graphs[1].L, L[0])
            self.set_parameter(variables, self.graphs[1].LA, LA[0])

        return logL


class SparseGPRegressionMeanVariancePrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, target_variables=None,
                 noise_free=True, diagonal_variance=True):
        super(SparseGPRegressionMeanVariancePrediction, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        self.noise_free = noise_free
        self.diagonal_variance = diagonal_variance

    def compute(self, F, variables):
        X = variables[self.model.X]
        N = X.shape[-2]
        Z = variables[self.model.inducing_inputs]
        noise_var = variables[self.model.noise_var]
        L = variables[self.graphs[1].L]
        LA = variables[self.graphs[1].LA]
        wv = variables[self.graphs[1].wv]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, Z, noise_var, L, LA, wv, kern_params = arrays_as_samples(
            F, [X, Z, noise_var, L, LA, wv, kern_params])

        Kxt = kern.K(F, Z, X, **kern_params)

        mu = F.linalg.gemm2(Kxt, wv, True, False)
        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            mu = mu + mean

        LinvKxt = F.linalg.trsm(L, Kxt)
        LAinvLinvKxt = F.linalg.trsm(LA, LinvKxt)

        if self.diagonal_variance:
            Ktt = kern.Kdiag(F, X, **kern_params)
            var = Ktt - F.sum(F.square(LinvKxt), axis=-2) + \
                F.sum(F.square(LAinvLinvKxt), axis=-2)
            if not self.noise_free:
                var += noise_var
        else:
            Ktt = kern.K(F, X, **kern_params)
            var = Ktt - F.linalg.syrk(LinvKxt, True) + \
                F.linalg.syrk(LAinvLinvKxt, True)
            if not self.noise_free:
                var += F.expand_dims(F.eye(N, dtype=X.dtype), axis=0) * \
                    F.expand_dims(noise_var, axis=-2)

        outcomes = {self.model.Y.uuid: (mu, var)}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class SparseGPRegressionSamplingPrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, rand_gen=None,
                 noise_free=True, diagonal_variance=True, jitter=0.):
        super(SparseGPRegressionSamplingPrediction, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        self.noise_free = noise_free
        self.diagonal_variance = diagonal_variance
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        self.jitter = jitter

    def compute(self, F, variables):
        X = variables[self.model.X]
        N = X.shape[-2]
        Z = variables[self.model.inducing_inputs]
        noise_var = variables[self.model.noise_var]
        L = variables[self.graphs[1].L]
        LA = variables[self.graphs[1].LA]
        wv = variables[self.graphs[1].wv]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, Z, noise_var, L, LA, wv, kern_params = arrays_as_samples(
            F, [X, Z, noise_var, L, LA, wv, kern_params])

        Kxt = kern.K(F, Z, X, **kern_params)

        mu = F.linalg.gemm2(Kxt, wv, True, False)
        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            mu = mu + mean

        LinvKxt = F.linalg.trsm(L, Kxt)
        LAinvLinvKxt = F.linalg.trsm(LA, LinvKxt)

        if self.diagonal_variance:
            Ktt = kern.Kdiag(F, X, **kern_params)
            var = Ktt - F.sum(F.square(LinvKxt), axis=-2) + \
                F.sum(F.square(LAinvLinvKxt), axis=-2)
            if not self.noise_free:
                var += noise_var
            die = self._rand_gen.sample_normal(
                shape=(self.num_samples,) + mu.shape[1:],
                dtype=self.model.F.factor.dtype)
            samples = mu + die * F.sqrt(F.expand_dims(var, axis=-1))
        else:
            Ktt = kern.K(F, X, **kern_params)
            cov = Ktt - F.linalg.syrk(LinvKxt, True) + \
                F.linalg.syrk(LAinvLinvKxt, True)
            if not self.noise_free:
                cov += F.eye(N, dtype=X.dtype) * noise_var
            if self.jitter > 0.:
                cov = cov + F.eye(cov.shape[-1], dtype=cov.dtype) * self.jitter
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


class SparseGPRegression(Module):
    """
    Variational sparse Gaussian process regression module

    This module contains a variational sparse Gaussian process with Gaussian likelihood.

    :param X: the input variables on which the random variables are conditioned.
    :type X: Variable
    :param kernel: the kernel of Gaussian process.
    :type kernel: Kernel
    :param noise_var: the variance of the Gaussian likelihood
    :type noise_var: Variable
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

    def __init__(self, X, kernel, noise_var, inducing_inputs=None,
                 inducing_num=10, mean_func=None,
                 rand_gen=None, dtype=None, ctx=None):
        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)
        if inducing_inputs is None:
            inducing_inputs = Variable(shape=(inducing_num, kernel.input_dim))
        inputs = [('X', X), ('inducing_inputs', inducing_inputs),
                  ('noise_var', noise_var)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(SparseGPRegression, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        self.mean_func = mean_func
        self.kernel = kernel

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
        Generate a model graph for sparse GP regression module.
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
            mean_func=self.mean_func, rand_gen=self._rand_gen, dtype=self.dtype,
            ctx=self.ctx)
        graph.F = ConditionalGaussianProcess.define_variable(
            X=graph.X, X_cond=graph.inducing_inputs, Y_cond=graph.U,
            kernel=self.kernel, shape=Y.shape, mean_func=self.mean_func,
            rand_gen=self._rand_gen, dtype=self.dtype, ctx=self.ctx)
        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(
            mean=graph.F, variance=graph.noise_var, rand_gen=self._rand_gen,
            dtype=self.dtype, ctx=self.ctx))
        graph.mean_func = self.mean_func
        graph.kernel = graph.U.factor.kernel
        post = Posterior(graph)
        # TODO: allow cloning kernel to be in both model and posterior.
        # post.F.assign_factor(ConditionalGaussianProcess(
        #     X=post.X, X_cond=post.inducing_inputs, Y_cond=post.U,
        #     kernel=self.kernel, mean_func=self.mean_func,
        #     rand_gen=self.rand_gen, dtype=self.dtype, ctx=self.ctx))
        # post.U.assign_factor(MultivariateNormal())
        post.L = Variable(shape=(M, M))
        post.LA = Variable(shape=(M, M))
        post.wv = Variable(shape=(M, Y.shape[-1]))
        return graph, [post]

    def _attach_default_inference_algorithms(self):
        """
        Attach the default inference algorithms for SparseGPRegression Module:
        log_pdf <- SparseGPRegressionLogPdf
        draw_samples <- ForwardSamplingAlgorithm
        prediction <- SparseGPRegressionMeanVariancePrediction
        """
        observed = [v for k, v in self.inputs] + \
            [v for k, v in self.outputs]
        self.attach_log_pdf_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SparseGPRegressionLogPdf(self._module_graph, self._extra_graphs[0], observed), alg_name='sgp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_draw_samples_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=ForwardSamplingAlgorithm(
                self._module_graph, observed),
            alg_name='sgp_sampling')

        self.attach_prediction_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SparseGPRegressionMeanVariancePrediction(
                self._module_graph, self._extra_graphs[0], observed),
            alg_name='sgp_predict')

    @staticmethod
    def define_variable(X, kernel, noise_var, shape=None, inducing_inputs=None,
                        inducing_num=10, mean_func=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a variable drawn from a sparse Gaussian process regression.

        :param X: the input variables on which the random variables are conditioned.
        :type X: Variable
        :param kernel: the kernel of Gaussian process.
        :type kernel: Kernel
        :param noise_var: the variance of the Gaussian likelihood
        :type noise_var: Variable
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
        gp = SparseGPRegression(
            X=X, kernel=kernel, noise_var=noise_var,
            inducing_inputs=inducing_inputs, inducing_num=inducing_num,
            mean_func=mean_func, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        rep = super(SparseGPRegression, self).replicate_self(attribute_map)

        rep.kernel = self.kernel.replicate_self(attribute_map)
        rep.mean_func = None if self.mean_func is None else self.mean_func.replicate_self(attribute_map)
        return rep
