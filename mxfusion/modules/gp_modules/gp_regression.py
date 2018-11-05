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
from ...components.distributions import GaussianProcess, Normal
from ...inference.inference_alg import SamplingAlgorithm
from ...components.distributions.random_gen import MXNetRandomGenerator
from ...util.inference import realize_shape
from ...inference.variational import VariationalInference
from ...util.customop import broadcast_to_w_samples
from ...components.variables.runtime_variable import arrays_as_samples


class GPRegressionLogPdf(VariationalInference):
    """
    The method to compute the logarithm of the probability density function of
    a Gaussian process model with Gaussian likelihood.
    """
    def compute(self, F, variables):
        X = variables[self.model.X]
        Y = variables[self.model.Y]
        noise_var = variables[self.model.noise_var]
        D = Y.shape[-1]
        N = X.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, Y, noise_var, kern_params = arrays_as_samples(
            F, [X, Y, noise_var, kern_params])

        K = kern.K(F, X, **kern_params) + \
            F.expand_dims(F.eye(N, dtype=X.dtype), axis=0) * \
            F.expand_dims(noise_var, axis=-2)
        L = F.linalg.potrf(K)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            Y = Y - mean
        LinvY = F.linalg.trsm(L, Y)
        logdet_l = F.linalg.sumlogdiag(F.abs(L))
        tmp = F.sum(F.reshape(F.square(LinvY) + np.log(2. * np.pi),
                              shape=(Y.shape[0], -1)), axis=-1)
        logL = - logdet_l * D - tmp/2

        with autograd.pause():
            self.set_parameter(variables, self.posterior.X, X[0])
            self.set_parameter(variables, self.posterior.L, L[0])
            self.set_parameter(variables, self.posterior.LinvY, LinvY[0])
        return logL


class GPRegressionSampling(SamplingAlgorithm):
    """
    The method for drawing samples from the prior distribution of a Gaussian process regression model.
    """
    def __init__(self, model, observed, num_samples=1, target_variables=None,
                 rand_gen=None):

        super(GPRegressionSampling, self).__init__(
            model=model, observed=observed, num_samples=num_samples,
            target_variables=target_variables)
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen

    def compute(self, F, variables):
        X = variables[self.model.X]
        noise_var = variables[self.model.noise_var]
        N = X.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, noise_var, kern_params = arrays_as_samples(
            F, [X, noise_var, kern_params])

        K = kern.K(F, X, **kern_params) + \
            F.expand_dims(F.eye(N, dtype=X.dtype), axis=0) * \
            F.expand_dims(noise_var, axis=-2)
        L = F.linalg.potrf(K)
        Y_shape = realize_shape(self.model.Y.shape, variables)
        out_shape = (self.num_samples,)+Y_shape

        L = broadcast_to_w_samples(F, L, out_shape[:-1] + out_shape[-2:-1])
        die = self._rand_gen.sample_normal(shape=out_shape,
                                           dtype=self.model.F.factor.dtype)
        y_samples = F.linalg.trmm(L, die)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            y_samples = y_samples + mean

        samples = {self.model.Y.uuid: y_samples}

        if self.target_variables:
            return tuple(samples[v] for v in self.target_variables)
        else:
            return samples


class GPRegressionMeanVariancePrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, noise_free=True,
                 diagonal_variance=True):
        super(GPRegressionMeanVariancePrediction, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        self.noise_free = noise_free
        self.diagonal_variance = diagonal_variance

    def compute(self, F, variables):
        X = variables[self.model.X]
        N = X.shape[-2]
        noise_var = variables[self.model.noise_var]
        X_cond = variables[self.graphs[1].X]
        L = variables[self.graphs[1].L]
        LinvY = variables[self.graphs[1].LinvY]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, noise_var, X_cond, L, LinvY, kern_params = arrays_as_samples(
            F, [X, noise_var, X_cond, L, LinvY, kern_params])

        Kxt = kern.K(F, X_cond, X, **kern_params)
        LinvKxt = F.linalg.trsm(L, Kxt)
        mu = F.linalg.gemm2(LinvKxt, LinvY, True, False)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            mu = mu + mean

        if self.diagonal_variance:
            Ktt = kern.Kdiag(F, X, **kern_params)
            var = Ktt - F.sum(F.square(LinvKxt), axis=-2)
            if not self.noise_free:
                var += noise_var
        else:
            Ktt = kern.K(F, X, **kern_params)
            var = Ktt - F.linalg.syrk(LinvKxt, True)
            if not self.noise_free:
                var += F.expand_dims(F.eye(N, dtype=X.dtype), axis=0) * \
                    F.expand_dims(noise_var, axis=-2)

        outcomes = {self.model.Y.uuid: (mu, var)}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class GPRegressionSamplingPrediction(SamplingAlgorithm):
    def __init__(self, model, posterior, observed, rand_gen=None,
                 noise_free=True, diagonal_variance=True, jitter=0.):
        super(GPRegressionSamplingPrediction, self).__init__(
            model=model, observed=observed, extra_graphs=[posterior])
        self.noise_free = noise_free
        self.diagonal_variance = diagonal_variance
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        self.jitter = jitter

    def compute(self, F, variables):
        X = variables[self.model.X]
        N = X.shape[-2]
        noise_var = variables[self.model.noise_var]
        X_cond = variables[self.graphs[1].X]
        L = variables[self.graphs[1].L]
        LinvY = variables[self.graphs[1].LinvY]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        X, noise_var, X_cond, L, LinvY, kern_params = arrays_as_samples(
            F, [X, noise_var, X_cond, L, LinvY, kern_params])

        Kxt = kern.K(F, X_cond, X, **kern_params)
        LinvKxt = F.linalg.trsm(L, Kxt)
        mu = F.linalg.gemm2(LinvKxt, LinvY, True, False)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            mu = mu + mean

        if self.diagonal_variance:
            Ktt = kern.Kdiag(F, X, **kern_params)
            var = Ktt - F.sum(F.square(LinvKxt), axis=-2)
            if not self.noise_free:
                var += noise_var
            die = self._rand_gen.sample_normal(
                shape=(self.num_samples,) + mu.shape[1:],
                dtype=self.model.F.factor.dtype)
            samples = mu + die * F.sqrt(F.expand_dims(var, axis=-1))
        else:
            Ktt = kern.K(F, X, **kern_params)
            cov = Ktt - F.linalg.syrk(LinvKxt, True)
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


class GPRegression(Module):
    """
    Gaussian process regression module

    This module contains a Gaussian process with Gaussian likelihood.

    :param X: the input variables on which the random variables are conditioned.
    :type X: Variable
    :param kernel: the kernel of Gaussian process.
    :type kernel: Kernel
    :param noise_var: the variance of the Gaussian likelihood
    :type noise_var: Variable
    :param mean_func: the mean function of Gaussian process.
    :type mean_func: MXFusionFunction
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """

    def __init__(self, X, kernel, noise_var, mean_func=None, rand_gen=None,
                 dtype=None, ctx=None):
        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)
        inputs = [('X', X), ('noise_var', noise_var)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(GPRegression, self).__init__(
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
        Generate a model graph for GP regression module.
        """
        Y = self.random_variable
        graph = Model(name='gp_regression')
        graph.X = self.X.replicate_self()
        graph.noise_var = self.noise_var.replicate_self()
        graph.F = GaussianProcess.define_variable(
            X=graph.X, kernel=self.kernel, shape=Y.shape,
            mean_func=self.mean_func, rand_gen=self._rand_gen,
            dtype=self.dtype, ctx=self.ctx)
        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(
            mean=graph.F, variance=graph.noise_var, rand_gen=self._rand_gen,
            dtype=self.dtype, ctx=self.ctx))
        graph.mean_func = self.mean_func
        graph.kernel = graph.F.factor.kernel
        # The posterior graph is used to store parameters for prediction
        post = Posterior(graph)
        post.L = Variable(shape=graph.X.shape[:-1]+graph.X.shape[-2:-1])
        post.LinvY = Variable(shape=graph.X.shape[:-1]+graph.Y.shape[-1:])
        post.X = Variable(shape=graph.X.shape)
        return graph, [post]

    def _attach_default_inference_algorithms(self):
        """
        Attach the default inference algorithms for GPRegression Module:
        log_pdf <- GPRegressionLogPdf
        draw_samples <- GPRegressionSampling
        prediction <- GPRegressionMeanVariancePrediction
        """
        observed = [v for k, v in self.inputs] + \
            [v for k, v in self.outputs]
        self.attach_log_pdf_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=GPRegressionLogPdf(self._module_graph, self._extra_graphs[0],
                                     observed),
            alg_name='gp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_draw_samples_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=GPRegressionSampling(self._module_graph, observed,
                                      rand_gen=self._rand_gen),
            alg_name='gp_sampling')

        self.attach_prediction_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=GPRegressionMeanVariancePrediction(
                self._module_graph, self._extra_graphs[0], observed),
            alg_name='gp_predict')

    @staticmethod
    def define_variable(X, kernel, noise_var, shape=None, mean_func=None,
                        rand_gen=None, dtype=None, ctx=None):
        """
        Creates and returns a variable drawn from a Gaussian process regression.

        :param X: the input variables on which the random variables are
        conditioned.
        :type X: Variable
        :param kernel: the kernel of Gaussian process
        :type kernel: Kernel
        :param noise_var: the variance of the Gaussian likelihood
        :type noise_var: Variable
        :param shape: the shape of the random variable(s) (the default shape is
        the same shape as *X* but the last dimension is changed to one.)
        :type shape: tuple or [tuple]
        :param mean_func: the mean function of Gaussian process
        :type mean_func: MXFusionFunction
        :param rand_gen: the random generator (default: MXNetRandomGenerator)
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context)
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """
        gp = GPRegression(
            X=X, kernel=kernel, noise_var=noise_var, mean_func=mean_func,
            rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        rep = super(GPRegression, self).replicate_self(attribute_map)

        rep.kernel = self.kernel.replicate_self(attribute_map)
        rep.mean_func = None if self.mean_func is None else self.mean_func.replicate_self(attribute_map)
        return rep
