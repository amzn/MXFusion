import numpy as np

from ...common.config import get_default_dtype
from ...components.distributions import GaussianProcess, Normal, ConditionalGaussianProcess
from ...components.distributions.random_gen import MXNetRandomGenerator
from ...components.functions.operators import broadcast_to
from ...components.variables.runtime_variable import arrays_as_samples
from ...components.variables.var_trans import PositiveTransformation
from ...components.variables.variable import Variable
from ...inference.inference_alg import SamplingAlgorithm
from ...inference.variational import VariationalInference
from ...models import Model, Posterior
from ...modules.module import Module
from ...util.customop import make_diagonal


# TODO: allow mean function in layers (original paper uses linear mean)


def multivariate_gaussian_kl_divergence(F, mean_1, cov_1, mean_2, cov_2):
    """
    computes KL(p_1 || p_2) where p1 and p2 are multivariate normals

    :param mean_1: M x D
    :param cov_1: D X M x M
    :param mean_2: M X D
    :param cov_2: D X M x M
    """

    M = mean_1.shape[-2]
    D = mean_1.shape[-1]

    L_1 = F.linalg.potrf(cov_1)  # D x M x M
    L_2 = F.linalg.potrf(cov_2)  # D x M x M

    mean_diff = mean_1 - mean_2  # M x D
    mean_diff = F.expand_dims(F.transpose(mean_diff), -1)

    LinvLs = F.linalg.trsm(L_2, L_1)  # D x M x M
    Linvmu = F.linalg.trsm(L_2, mean_diff)  # D x M x M

    return 0.5 * (2 * F.linalg.sumlogdiag(L_2).sum() -
                  2 * F.linalg.sumlogdiag(L_1).sum() +
                  F.sum(F.square(LinvLs)) +
                  F.sum(F.square(Linvmu)) -
                  M * D)


def compute_marginal_f(F, L, Kfu, Kff_diag, mu_u, Ls, mean_prior_u=None, mean_prior_f=None):
    """
    Compute q(f) parameters

    :param F: mx.nd of mx.sym
    :param L: cholesky(K_uu)
    :param Kfu: k(x, z)
    :param Kff_diag: diag(k(x, x))
    :param mu_u: mean of variational distribution q(u)
    :param Ls: cholesky of variational distribution q(u)
    :param mean_prior_u: gp mean function evaluated at inducing point locations
    :param mean_prior_f: gp mean function evaluated at x locations

    :return: mean and variance of q(f|u)
    """
    Kmmi = F.linalg.potri(L)

    mu_u, Kmmi, Kfu = arrays_as_samples(F, [mu_u, Kmmi, Kfu])

    KfuKmmi = F.linalg.gemm2(Kfu, Kmmi, alpha=1.0)

    if mean_prior_u is not None and mean_prior_f is not None:
        mu_f = mean_prior_f + F.linalg.gemm2(mu_u - mean_prior_u, KfuKmmi, alpha=1.0)
    else:
        mu_f = F.linalg.gemm2(KfuKmmi, mu_u, alpha=1.0)

    aux = F.expand_dims(Kff_diag - F.sum(KfuKmmi * Kfu, -1), -1)

    S = Kfu.shape[0]
    N = Kfu.shape[1]
    D = Ls.shape[1]

    # TODO: work out how to vectorize this...
    tmp = F.zeros((S, N, D), dtype='float64')  ## FIXME: dype
    for i in range(D):
        tmp[:, :, i] = F.sum(F.square(F.linalg.gemm2(KfuKmmi, Ls[:, i, :, :], transpose_b=True)), -1)

    v_f = aux + tmp
    return mu_f, v_f


def gaussian_variational_expectation(F, y, variance, f_mean, f_var):
    """

    :param F: mx.nd or mx.sym
    :param y: observed y
    :param variance: likelihood variance
    :param f_mean: prediction mean
    :param f_var: prediction variance
    """
    return -0.5 * np.log(2 * np.pi) - 0.5 * F.log(variance) \
           - 0.5 * (F.square(y - f_mean) + f_var) / variance


def dgp_prediction(F, variables, model, posterior, n_layers, random_generator, n_samples, jitter, dtype):
    """
    Predict from final layer of DGP. Returns both posterior samples and mean and variance of posterior samples

    :param F: mx.nd or mx.sym
    :param variables: variables dictionary
    :param model: model graph
    :param posterior: posterior graph
    :param n_layers: number of layers
    :param random_generator: random number generator used to draw samples at each layer
    :param n_samples: number of samples to propagate through layers
    :param jitter: value of jitter to add to k_uu for numerical stability
    :param dtype: dtype of samples
    :return: (samples, mean, variance)
    """
    x = variables[model.X]

    samples = x

    for layer in range(n_layers):
        # Get layer specific variables
        z = variables[getattr(model, 'inducing_inputs_' + str(layer))]
        S_W = variables[getattr(posterior, 'qU_cov_W_' + str(layer))]
        S_diag = variables[getattr(posterior, 'qU_cov_diag_' + str(layer))]
        mu = variables[getattr(posterior, 'qU_mean_' + str(layer))]
        kern = getattr(model, 'kern_' + str(layer))
        kern_params = kern.fetch_parameters(variables)

        # Compute covariance matrix from parts
        cov = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)

        _, M, D = mu.shape
        k_uu = kern.K(F, z, **kern_params) + F.eye(M, dtype=dtype) * jitter
        L = F.linalg.potrf(k_uu)
        Kff_diag = kern.Kdiag(F, x, **kern_params)

        Ls = F.linalg.potrf(cov)
        samples, Ls, z_s, Kff_diag = arrays_as_samples(F, [samples, Ls, z, Kff_diag])

        k_xu = kern.K(F, samples, z_s, **kern_params)

        # find q(f) from q(u)
        mu_f, v_f = compute_marginal_f(F, L, k_xu, Kff_diag, mu, Ls)

        # sample from q(f)
        L_f = F.sqrt(v_f)
        die = random_generator.sample_normal(shape=(n_samples, x.shape[1], D), dtype=dtype)

        samples = F.broadcast_add(F.broadcast_mul(L_f, die), mu_f)

    return samples, mu_f, v_f


class DeepGPLogPdf(VariationalInference):
    """
    The inference algorithm for computing the variational lower bound of the stochastic variational Gaussian process
    with Gaussian likelihood.
    """

    def __init__(self, model, posterior, observed, n_layers, jitter=1e-6, n_samples=10, dtype=None):
        """
        :param model: Model graph
        :param posterior: Posterior graph
        :param observed: List of observed variables
        :param jitter: Jitter for numerical stability, defaults to 1e-6
        """

        super().__init__(model=model, posterior=posterior, observed=observed)

        if dtype is None:
            self.dtype = get_default_dtype()
        else:
            self.dtype = dtype

        self.log_pdf_scaling = 1.
        self.jitter = jitter
        self.n_layers = n_layers
        if self.n_layers <= 1:
            raise ValueError('Number of layers must be greater than one')
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples

    def compute(self, F, variables):
        """
        Compute ELBO of model

        :param F:
        :param variables:
        :return:
        """
        y = variables[self.model.Y]

        # Compute model fit term
        _, mu_f, v_f = dgp_prediction(F, variables, self.model, self.posterior, self.n_layers,
                                      self._rand_gen, self.n_samples, self.jitter, self.dtype)
        noise_var = variables[self.model.noise_var]

        self.v_f_mean = F.mean(v_f, 0).asnumpy()
        self.mu_f_mean = F.mean(mu_f, 0).asnumpy()
        lik = gaussian_variational_expectation(F, y, noise_var, mu_f, v_f)

        # Compute kl term
        kl = self._compute_kl(F, variables)

        # TODO: remove this, it's for debugging
        self.kl = kl.asnumpy()
        self.lik = lik.mean(axis=0).asnumpy()

        return self.log_pdf_scaling * lik.mean(axis=0).sum() - kl

    def _compute_kl(self, F, variables):
        """
        Compute sum of KL divergences for each layer of DGP
        """
        kl = 0
        for layer in range(self.n_layers):
            # Get layer specific variables
            z = variables[getattr(self.model, 'inducing_inputs_' + str(layer))]
            S_W = variables[getattr(self.posterior, 'qU_cov_W_' + str(layer))]
            S_diag = variables[getattr(self.posterior, 'qU_cov_diag_' + str(layer))]
            mu = variables[getattr(self.posterior, 'qU_mean_' + str(layer))]
            kern = getattr(self.model, 'kern_' + str(layer))
            kern_params = kern.fetch_parameters(variables)

            # Compute covariance matrix from parts
            cov = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)

            _, M, D = mu.shape
            k_uu = kern.K(F, z, **kern_params) + F.eye(M, dtype=self.dtype) * self.jitter
            k_uu = F.tile(k_uu, (1, D, 1, 1))

            mu = F.reshape(mu, mu.shape[1:])
            cov = F.reshape(cov, cov.shape[1:])
            k_uu = F.reshape(k_uu, k_uu.shape[1:])
            kl = kl + multivariate_gaussian_kl_divergence(F, mu, cov, F.zeros((M, D), dtype=self.dtype), k_uu)
        return kl


class DeepGPMeanVariancePrediction(SamplingAlgorithm):
    """
    Calculates mean and variance of deep GP prediction
    """
    def __init__(self, model, posterior, observed, n_layers, noise_free=True, n_samples=10, jitter=1e-6, dtype=None):

        super().__init__(model=model, observed=observed, extra_graphs=[posterior])
        if dtype is None:
            self.dtype = get_default_dtype()
        else:
            self.dtype = dtype

        self.jitter = jitter
        self.noise_free = noise_free
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples

    def compute(self, F, variables):
        _, mu_f, v_f = dgp_prediction(F, variables, self.model, self._extra_graphs[0], self.n_layers, self._rand_gen,
                                      self.n_samples, self.jitter, self.dtype)

        # Add likelihood noise if required
        noise_var = variables[self.model.noise_var]
        if not self.noise_free:
            v_f = v_f + noise_var

        mean = mu_f.mean(axis=0)

        variance_of_mean = F.mean(F.square(mu_f - mean), axis=0)
        var = variance_of_mean + v_f.mean(axis=0)

        outcomes = {self.model.Y.uuid: (mean, var)}

        if self.target_variables:
            return tuple(outcomes[v] for v in self.target_variables)
        else:
            return outcomes


class DeepGPForwardSampling(SamplingAlgorithm):
    """
    Calculates mean and variance of deep GP prediction
    """
    def __init__(self, model, posterior, observed, n_layers, noise_free=True, n_samples=10, jitter=1e-6, dtype=None):

        super().__init__(model=model, observed=observed, extra_graphs=[posterior])
        if dtype is None:
            self.dtype = get_default_dtype()
        else:
            self.dtype = dtype
        self.jitter = jitter
        self.noise_free = noise_free
        self.n_layers = n_layers
        self._rand_gen = MXNetRandomGenerator
        self.n_samples = n_samples

    def compute(self, F, variables):
        samples, _, _ = dgp_prediction(F, variables, self.model, self._extra_graphs[0], self.n_layers,
                                       self._rand_gen, self.n_samples, self.jitter, self.dtype)

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
                 num_inducing=10, mean=None, n_samples=10, dtype=None, ctx=None):
        """
        :param X: Input variable
        :param kernels: List of kernels for each layer
        :param noise_var: Noise variance for likelihood at final layer
        :param inducing_inputs: List of variables that represent the inducing points at each layer or None
        :param num_inducing: Number of inducing points at each layer in inducing_inputs is None
        :param mean: Not used yet
        :param dtype: dtype to use when creating mxnet arrays
        :param ctx: mxnet context
        """

        self.n_layers = len(kernels)

        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)

        self.layer_input_dims = [kern.input_dim for kern in kernels]
        self.layer_output_dims = self.layer_input_dims[1:] + [1]

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
        self.mean_func = mean
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
                kernel=self.kernels[i], shape=(N, self.layer_output_dims[i]), mean=None,
                rand_gen=self._rand_gen, dtype=self.dtype, ctx=self.ctx)
            setattr(graph, 'F_' + str(i), f)

            setattr(graph, 'kern_' + str(i), u.factor.kernel)

            setattr(post, 'qU_cov_diag_' + str(i), Variable(shape=(self.layer_output_dims[i], M,),
                                                            transformation=PositiveTransformation()))
            setattr(post, 'qU_cov_W_' + str(i), Variable(shape=(self.layer_output_dims[i], M, M)))
            setattr(post, 'qU_mean_' + str(i), Variable(shape=(M, self.layer_output_dims[i])))

        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(mean=getattr(graph, 'F_' + str(self.n_layers - 1)),
                                 variance=broadcast_to(graph.noise_var, graph.Y.shape),
                                 rand_gen=self._rand_gen,
                                 dtype=self.dtype, ctx=self.ctx))

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
                                                              self.n_layers, n_samples=self.n_samples,
                                                              dtype=self.dtype), alg_name='dgp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_prediction_algorithms(
           targets=self.output_names, conditionals=self.input_names,
           algorithm=DeepGPMeanVariancePrediction(self._module_graph, self._extra_graphs[0], observed, self.n_layers,
                                                  n_samples=self.n_samples, dtype=self.dtype), alg_name='dgp_predict')

        self.attach_draw_samples_algorithms(
           targets=self.output_names, conditionals=self.input_names,
           algorithm=DeepGPForwardSampling(self._module_graph, self._extra_graphs[0], observed, self.n_layers,
                                           n_samples=self.n_samples, dtype=self.dtype), alg_name='dgp_sample')

    @staticmethod
    def define_variable(X, kernels, noise_var, shape=None, inducing_inputs=None, num_inducing=10, mean_func=None,
                        n_samples=10, dtype=None, ctx=None):
        """
        Creates and returns a variable drawn from a doubly stochastic deep GP

        :param X: Input variable
        :param kernels: List of kernels for each layer
        :param noise_var: Noise variance for likelihood at final layer
        :param shape: Shape of variable
        :param inducing_inputs: List of variables that represent the inducing points at each layer or None
        :param num_inducing: Number of inducing points at each layer in inducing_inputs is None
        :param mean_func: Not used yet
        :param dtype: dtype to use when creating mxnet arrays
        :param ctx: mxnet context
        """
        gp = DeepGPRegression(
            X=X, kernels=kernels, noise_var=noise_var,
            inducing_inputs=inducing_inputs, num_inducing=num_inducing,
            mean=mean_func, n_samples=n_samples, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        rep = super().replicate_self(attribute_map)

        rep.kernels = [k.replicate_self(attribute_map) for k in self.kernels]
        rep.mean_func = None if self.mean_func is None else self.mean_func.replicate_self(attribute_map)
        return rep
