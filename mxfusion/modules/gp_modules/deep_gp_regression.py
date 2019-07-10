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

import mxnet as mx
import numpy as np

from ...common.config import get_default_dtype
from ...components.distributions import ConditionalGaussianProcess, GaussianProcess, Normal
from ...components.distributions.random_gen import MXNetRandomGenerator
from ...components.functions.operators import broadcast_to
from ...components.variables.runtime_variable import broadcast_sample_dimension, as_samples
from ...components.variables.var_trans import PositiveTransformation
from ...components.variables.variable import Variable
from ...inference.inference_alg import SamplingAlgorithm
from ...inference.variational import VariationalInference
from ...models import Model, Posterior
from ...modules.module import Module
from ...runtime.distributions.multivariate_normal import MultivariateNormalRuntime
from ...util.customop import make_diagonal


def gaussian_variational_expectation(F, y, variance, f_mean, f_var):
    """
    :param F: the execution context (mxnet.ndarray or mxnet.symbol)
    :type F: Python module
    :param y: observed y of shape (n_samples, n_data, n_output_dims)
    :type y: MXNet NDArray
    :param variance: likelihood variance
    :type variance: MXNet NDArray
    :param f_mean: prediction mean of shape (n_samples, n_output_dims, n_data)
    :type variance: MXNet NDArray
    :param f_var: prediction variance of shape (n_samples, n_output_dims, n_data)
    :type variance: MXNet NDArray
    :return: Likelihood of data of shape (n_samples, n_data)
    :rtype: MXNet NDArray
    """
    y = F.transpose(y, (0, 2, 1))
    return -0.5 * np.log(2 * np.pi) - 0.5 * F.log(variance) \
           - 0.5 * (F.square(y - f_mean) + f_var) / variance


def predict_from_layer(F, inputs, layer, variables, posterior, model, jitter, mean):
    """
    Computes p(f) for a particular layer given the x locations to predict at

    :param mean:
    :param F: the execution context (mxnet.ndarray or mxnet.symbol)
    :type F: Python module
    :param inputs: Input locations to predict at
    :type inputs: MXNet NDArray
    :param layer: Zero based layer index to predict from
    :type layer: int
    :param variables: the set of MXNet arrays that holds the values of
    variables at runtime.
    :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
    :param posterior: Graph of q(u)
    :type posterior: mxfusion.models.Posterior
    :param model: Factor graph of entire DGP model
    :type model: mxfusion.models.Model
    :param jitter: Level of jitter for numerical stability
    :type jitter: float
    :return: Multivariate normal distribution describing p(f)
    :rtype: mxfusion.runtime.distributions.MultivariateNormalRuntime
    """
    q_u = get_q_u(F, layer, posterior, variables)

    # Get p(f|u) object
    z = variables[getattr(model, 'inducing_inputs_' + str(layer))]
    conditional_gp = getattr(model, 'F_' + str(layer))
    kern = getattr(model, 'kern_' + str(layer))
    kern_params = kern.fetch_parameters(variables)
    mu = variables[getattr(posterior, 'qU_mean_' + str(layer))]
    rv_shape = mu.shape[1:]

    samples, z_w_samples = broadcast_sample_dimension([inputs, z])

    if mean is not None:
        kern_params['mean'] = F.expand_dims(F.transpose(mean(samples), (0, 2, 1)), -1)

    q_u = MultivariateNormalRuntime(q_u.mean, q_u.covariance)
    return conditional_gp.factor.marginalise_conditional_distribution(samples, z, q_u, rv_shape[0], jitter, True,
                                                                      **kern_params)


def propagate_samples_through_layers(F, variables, model, posterior, n_layers, n_samples, jitter, mean_functions,
                                     final_layer=None, start_layer=None):
    """
    Propagates samples from start_layer to final_layer

    :param mean_functions:
    :param F: the execution context (mxnet.ndarray or mxnet.symbol)
    :type F: Python module
    :param variables: the set of MXNet arrays that holds the values of
    variables at runtime.
    :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
    :param model: Factor graph of entire DGP model
    :type model: mxfusion.models.Model
    :param posterior: Graph of q(u)
    :type posterior: mxfusion.models.Posterior
    :param n_layers: number of layers in model
    :type n_layers: int
    :param n_samples: number of samples to propagate through layers
    :type n_samples: int
    :param jitter: Level of jitter to add to diagonal of covariance
    :type jitter: float
    :param final_layer: Zero-based index of last layer to propagate samples through
    :type final_layer: int
    :param start_layer: Zero-based index of first layer to propagate samples through
    :type start_layer: int
    :return: posterior at specified layer, contains (samples, mean, variance)
    :rtype: MXNet NDArray
    """

    if final_layer is None:
        final_layer = n_layers

    if start_layer is None:
        start_layer = 0

    x = variables[model.X]

    samples = x

    for layer in range(start_layer, final_layer):
        # Predict from layer
        p_f = predict_from_layer(F, samples, layer, variables, posterior, model, jitter, mean_functions[layer])

        # Draw samples
        samples = p_f.draw_samples(n_samples)
        samples = F.transpose(samples, (0, 2, 1))

    return samples, p_f.mean, p_f.variance


def get_q_u(F, layer, posterior, variables):
    """
    Makes a MultivariateNormal run time distribution for the variational distribution at the inducing points.
    """
    # Make q(u) multivariate normal object
    S_W = variables[getattr(posterior, 'qU_cov_W_' + str(layer))]
    S_diag = variables[getattr(posterior, 'qU_cov_diag_' + str(layer))]
    cov = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)
    mu = variables[getattr(posterior, 'qU_mean_' + str(layer))]
    return MultivariateNormalRuntime(mu, cov)


class DeepGPLogPdf(VariationalInference):
    """
    The inference algorithm for computing the variational lower bound of the stochastic variational Gaussian process
    with Gaussian likelihood.
    """

    def __init__(self, model, posterior, observed, n_layers, mean_functions, jitter=1e-6, n_samples=5, dtype=None):
        """
        :param model: Factor graph of entire DGP model
        :type model: mxfusion.models.Model
        :param posterior: Graph of q(u)
        :type posterior: mxfusion.models.Posterior
        :param observed: List of observed variables
        :type observed: List[mxfusion.components.Variable]
        :param n_layers: Number of layers in model
        :type n_layers: int
        :param jitter: Jitter for numerical stability, defaults to 1e-6
        :type jitter: float
        :param n_samples: Number of samples to use to estimate the lower bound
        :type n_samples: int
        """

        super().__init__(model=model, posterior=posterior, observed=observed)

        if dtype is None:
            self.dtype = get_default_dtype()
        else:
            self.dtype = dtype

        self.log_pdf_scaling = 1.
        self.jitter = jitter
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples
        self.mean_functions = mean_functions

    def compute(self, F, variables):
        """
        Compute ELBO of model

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :return: Estimator of the ELBO
        :rtype: MXNet NDArray
        """
        y = variables[self.model.Y]

        # Compute model fit term
        _, mu_f, v_f = propagate_samples_through_layers(F, variables, self.model, self.posterior, self.n_layers,
                                                        self.n_samples, self.jitter, self.mean_functions)
        noise_var = variables[self.model.noise_var]

        lik = gaussian_variational_expectation(F, y, noise_var, mu_f, v_f)

        # Compute kl term
        kl = self._compute_kl(F, variables)
        return self.log_pdf_scaling * lik.mean(axis=0).sum() - kl

    def _compute_kl(self, F, variables):
        """
        Compute sum of KL divergences for each layer of DGP
        """
        kl = 0
        for layer in range(self.n_layers):
            # build variational multivariate normal distribution
            q_u = get_q_u(F, layer, self.posterior, variables)

            # TODO: use GP runtime distribution to extract GP distribution when available
            # build prior multivariate normal distribution
            z = variables[getattr(self.model, 'inducing_inputs_' + str(layer))]
            kern = getattr(self.model, 'kern_' + str(layer))
            kern_params = kern.fetch_parameters(variables)
            gp_dist = getattr(self.model, 'U_' + str(layer))
            n_output_dims = q_u.mean.shape[1]
            rv_shape = (n_output_dims, z.shape[1])
            p_u = gp_dist.factor.get_multivariate_normal(z, rv_shape, self.jitter, **kern_params)

            # calculate kl for this layer
            kl = kl + q_u.kl_divergence(p_u).sum()
        return kl


class DeepGPMeanVariancePrediction(SamplingAlgorithm):
    """
    Calculates mean and variance of deep GP prediction
    """

    def __init__(self, model, posterior, observed, n_layers, mean_functions, noise_free=True, jitter=1e-6, n_samples=10):
        """
        :param model: Factor graph of entire DGP model
        :type model: mxfusion.models.Model
        :param posterior: Graph of q(u)
        :type posterior: mxfusion.models.Posterior
        :param observed: List of observed variables
        :type observed: List[mxfusion.components.Variable]
        :param n_layers: Number of layers in model
        :type n_layers: int
        :param noise_free: Whether to add the liklihood variance to the posterior variance
        :type noise_free: bool
        :param jitter: Jitter for numerical stability, defaults to 1e-6
        :type jitter: float
        :param n_samples: Number of samples to use to estimate the lower bound
        :type n_samples: int
        """

        super().__init__(model=model, observed=observed, extra_graphs=[posterior])

        self.jitter = jitter
        self.noise_free = noise_free
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples
        self.prediction_layer = n_layers
        self.start_layer = 0
        self.mean_functions = mean_functions

    def compute(self, F, variables):
        """
        Compute predictive mean and variance

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :return: Estimator of the ELBO
        :rtype: MXNet NDArray
        """
        _, mu_f, v_f = propagate_samples_through_layers(F, variables, self.model, self._extra_graphs[0], self.n_layers,
                                                        self.n_samples, self.jitter, self.mean_functions, self.prediction_layer,
                                                        self.start_layer)

        # Add likelihood noise if required
        noise_var = variables[self.model.noise_var]
        if not self.noise_free:
            v_f = v_f + noise_var

        var = F.square(mu_f - mu_f.mean(axis=0, keepdims=True)) + v_f

        outcomes = {self.model.Y.uuid: (F.transpose(mu_f, (0, 2, 1)), F.transpose(var, (0, 2, 1)))}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class DeepGPRegressionSamplingPrediction(SamplingAlgorithm):
    """
    Calculates mean and variance of deep GP prediction
    """

    def __init__(self, model, posterior, observed, n_layers, mean_functions, noise_free=True, jitter=1e-6, n_samples=10):
        """
        :param model: Factor graph of entire DGP model
        :type model: mxfusion.models.Model
        :param posterior: Graph of q(u)
        :type posterior: mxfusion.models.Posterior
        :param observed: List of observed variables
        :type observed: List[mxfusion.components.Variable]
        :param n_layers: Number of layers in model
        :type n_layers: int
        :param noise_free: Whether to add the liklihood variance to the posterior variance
        :type noise_free: bool
        :param jitter: Jitter for numerical stability, defaults to 1e-6
        :type jitter: float
        :param n_samples: Number of samples to use to estimate the lower bound
        :type n_samples: int
        """
        super().__init__(model=model, observed=observed, extra_graphs=[posterior])
        self.jitter = jitter
        self.noise_free = noise_free
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples
        self.mean_functions = mean_functions

    def compute(self, F, variables):
        """
        Draw samples from predictive distribution

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :return: Estimator of the ELBO
        :rtype: MXNet NDArray
        """

        samples, _, _ = propagate_samples_through_layers(F, variables, self.model, self.posterior, self.n_layers,
                                                         self.n_samples, self.jitter, self.mean_functions)

        # Add likelihood noise if required
        noise_var = variables[self.model.noise_var]
        if not self.noise_free:
            samples = samples + self._rand_gen.sample_normal(0, F.sqrt(noise_var), shape=samples.shape)

        outcomes = {self.model.Y.uuid: samples}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class DeepGPRegression(Module):
    """
    Deep Gaussian process using doubly stochastic variational inference from:
    Doubly Stochastic Variational Inference for Deep Gaussian Processes (Hugh Salimbeni, Marc Deisenroth)
    https://arxiv.org/abs/1705.08933
    """

    def __init__(self, X, kernels, noise_var, inducing_inputs=None,
                 num_inducing=10, n_outputs=1, mean_functions=None, n_samples=5, dtype=None, ctx=None):
        """
        :param X: Input variable
        :type X: mxfusion.components.Variable
        :param kernels: List of kernels for each layer
        :type kernels: List[mxfusion.components.distributions.gp.kernels.kernel.Kernel]
        :param noise_var: Noise variance for likelihood at final layer
        :type noise_var: mxfusion.components.Variable
        :param inducing_inputs: List of variables that represent the inducing points at each layer or None
        :type inducing_inputs: List[mxfusion.components.Variable]
        :param num_inducing: Number of inducing points at each layer in inducing_inputs is None
        :type num_inducing: int
        :param n_outputs: Dimensionality of outputs
        :type n_outputs: int
        :param mean: List of mean functions for use at each layer
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """

        self.n_layers = len(kernels)

        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)

        self.layer_input_dims = [kern.input_dim for kern in kernels]
        self.layer_output_dims = self.layer_input_dims[1:] + [n_outputs]

        if inducing_inputs is None:
            inducing_inputs = [Variable(shape=(num_inducing, self.layer_input_dims[i])) for i in range(self.n_layers)]

        self.inducing_inputs = inducing_inputs
        inducing_inputs_tuples = []

        for i, inducing in enumerate(inducing_inputs):
            inducing_inputs_tuples.append(('inducing_inputs_' + str(i), inducing))

        inputs = [('X', X)] + inducing_inputs_tuples + [('noise_var', noise_var)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super().__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, dtype=dtype, ctx=ctx)
        self.mean_functions = [None] * self.n_layers if mean_functions is None else mean_functions
        self.kernels = kernels
        self.n_samples = n_samples

    def _generate_outputs(self, output_shapes=None):
        """
        Generate the output of the module with given output_shapes.
        :param output_shapes: the shapes of all the output variables
        :type output_shapes: {str: tuple}
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

        graph = Model(name='dgp_regression')
        graph.X = self.X.replicate_self()
        graph.noise_var = self.noise_var.replicate_self()

        post = Posterior(graph)
        N = Y.shape[0]

        for i in range(self.n_layers):
            z = self.inducing_inputs[i].replicate_self()
            setattr(graph, 'inducing_inputs_' + str(i), z)

            M = z.shape[0]

            u = GaussianProcess.define_variable(X=z, kernel=self.kernels[i], shape=(M, self.layer_output_dims[i]),
                                                rand_gen=self._rand_gen, dtype=self.dtype,
                                                ctx=self.ctx)
            setattr(graph, 'U_' + str(i), u)
            f = ConditionalGaussianProcess.define_variable(
                X=graph.X, X_cond=z, Y_cond=u,
                kernel=self.kernels[i], shape=(N, self.layer_output_dims[i]),
                rand_gen=self._rand_gen, dtype=self.dtype, ctx=self.ctx)
            setattr(graph, 'F_' + str(i), f)

            setattr(graph, 'kern_' + str(i), u.factor.kernel)

            setattr(post, 'qU_cov_diag_' + str(i), Variable(shape=(self.layer_output_dims[i], M,),
                                                            transformation=PositiveTransformation()))
            setattr(post, 'qU_cov_W_' + str(i), Variable(shape=(self.layer_output_dims[i], M, M)))
            setattr(post, 'qU_mean_' + str(i), Variable(shape=(self.layer_output_dims[i], M)))

        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(mean=getattr(graph, 'F_' + str(self.n_layers - 1)),
                                 variance=broadcast_to(graph.noise_var, graph.Y.shape)))

        return graph, [post]

    def _attach_default_inference_algorithms(self):
        """
        Attach the default inference algorithms for SVGPRegression Module:
        log_pdf <- DeepGPLogPdf
        prediction <- DeepGPMeanVariancePrediction
        """
        observed = [v for k, v in self.inputs] + \
                   [v for k, v in self.outputs]
        self.attach_log_pdf_algorithms(targets=self.output_names, conditionals=self.input_names,
                                       algorithm=DeepGPLogPdf(self._module_graph, self._extra_graphs[0], observed,
                                                              self.n_layers, self.mean_functions, n_samples=self.n_samples,
                                                              dtype=self.dtype), alg_name='dgp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_prediction_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=DeepGPMeanVariancePrediction(self._module_graph, self._extra_graphs[0], observed, self.n_layers,
                                                   self.mean_functions, n_samples=self.n_samples), alg_name='dgp_predict')

        self.attach_draw_samples_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=DeepGPRegressionSamplingPrediction(self._module_graph, self._extra_graphs[0], observed,
                                                         self.n_layers, self.mean_functions, n_samples=self.n_samples),
            alg_name='dgp_sample')

    @staticmethod
    def define_variable(X, kernels, noise_var, shape=None, inducing_inputs=None, num_inducing=10, mean_functions=None,
                        n_samples=10, n_outputs=1, dtype=None, ctx=None):
        """
        :param X: Input variable
        :type X: mxfusion.components.Variable
        :param kernels: List of kernels for each layer
        :type kernels: List[mxfusion.components.distributions.gp.kernels.kernel.Kernel]
        :param noise_var: Noise variance for likelihood at final layer
        :type noise_var: mxfusion.components.Variable
        :param shape: Shape of the random variable output
        :type shape: Tuple
        :param inducing_inputs: List of variables that represent the inducing points at each layer or None
        :type inducing_inputs: List[mxfusion.components.Variable]
        :param num_inducing: Number of inducing points at each layer in inducing_inputs is None
        :type num_inducing: int
        :param n_outputs: Dimensionality of outputs
        :type n_outputs: int
        :param mean_func: Not used yet
        :param n_samples: Number of samples to use in inference algorithms
        :type n_samples: int
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """
        gp = DeepGPRegression(
            X=X, kernels=kernels, noise_var=noise_var,
            inducing_inputs=inducing_inputs, num_inducing=num_inducing,
            mean_functions=mean_functions, n_outputs=n_outputs, n_samples=n_samples, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        rep = super().replicate_self(attribute_map)

        rep.kernels = [k.replicate_self(attribute_map) for k in self.kernels]
        rep.mean_functions = None if self.mean_functions is None else self.mean_functions
        return rep
