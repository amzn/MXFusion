import numpy as np
from mxnet import autograd
from ..module import Module
from ...models import Model, Posterior
from ...components.variables.variable import Variable
from ...components.variables.var_trans import PositiveTransformation
from ...components.distributions import GaussianProcess, Normal, ConditionalGaussianProcess, MultivariateNormal
from ...inference.variational import VariationalInference, VariationalSamplingAlgorithm
from ...inference.forward_sampling import ForwardSamplingAlgorithm
from ...util.customop import make_diagonal


class SVGPRegr_log_pdf(VariationalInference):
    def __init__(self, model, posterior, observed, jitter=0.):
        super(SVGPRegr_log_pdf, self).__init__(
            model=model, posterior=posterior, observed=observed)
        self.log_pdf_scaling = 1
        self.jitter = jitter

    def compute(self, F, variables):
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

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter
        Kuf = kern.K(F, Z, X, **kern_params)
        Kff_diag = kern.Kdiag(F, X, **kern_params)

        S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            Y = Y - mean

        psi1Y = F.linalg.gemm2(Kuf, Y, False, False)
        L = F.linalg.potrf(Kuu)
        Ls = F.linalg.potrf(S)
        LinvLs = F.linalg.trsm(L, Ls)
        Linvmu = F.linalg.trsm(L, mu)
        LinvKuf = F.linalg.trsm(L, Kuf)

        LinvKufY = F.linalg.trsm(L, psi1Y)/noise_var
        LmInvPsi2LmInvT = F.linalg.syrk(LinvKuf)/noise_var
        LinvSLinvT = F.linalg.syrk(LinvLs)
        LmInvSmuLmInvT = LinvSLinvT*D + F.linalg.syrk(Linvmu)

        KL_u = (M/2. + F.linalg.sumlogdiag(Ls))*D - F.linalg.sumlogdiag(L)*D\
            - F.sum(F.sum(F.square(LinvLs), axis=-1), axis=-1)/2.*D \
            - F.sum(F.sum(F.square(Linvmu), axis=-1), axis=-1)/2.

        logL = -F.sum(F.sum(F.square(Y)/noise_var + np.log(2. * np.pi) +
                            F.log(noise_var), axis=-1), axis=-1)/2.
        logL = logL - D/2.*F.sum(Kff_diag, axis=-1)/noise_var
        logL = logL - F.sum(F.sum(LmInvSmuLmInvT*LmInvPsi2LmInvT, axis=-1),
                            axis=-1)/2.
        logL = logL + F.sum(F.sum(F.square(LinvKuf)/noise_var, axis=-1),
                            axis=-1)*D/2.
        logL = logL + F.sum(F.sum(Linvmu*LinvKufY, axis=-1), axis=-1)
        logL = logL + self.model.U.factor.log_pdf_scaling*KL_u
        return logL


# class SVGPRegr_draw_samples_independent(VariationalSamplingAlgorithm):
#     def __init__(self, model, posterior, observed, num_samples=1,
#                  target_variables=None, jitter=0.):
#         super(SVGPRegr_draw_samples_independent, self).__init__(
#             model=model, posterior=posterior, observed=observed,
#             num_samples=num_samples, target_variables=target_variables)
#         self.jitter = jitter
#
#     def compute(self, F, variables):
#         X = variables[self.model.X]
#         Z = variables[self.model.inducing_inputs]
#         noise_var = variables[self.model.noise_var]
#         mu = variables[self.posterior.qU_mean]
#         S_W = variables[self.posterior.qU_cov_W]
#         S_diag = variables[self.posterior.qU_cov_diag]
#         M = Z.shape[-2]
#         kern = self.model.kernel
#         kern_params = kern.fetch_parameters(variables)
#
#         S = F.linalg.syrk(S_W) + make_diagonal(F, S_diag)
#
#         Kuu = kern.K(F, Z, **kern_params)
#         if self.jitter > 0.:
#             Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter
#
#         L = F.linalg.potrf(Kuu)
#         Ls = F.linalg.potrf(S)
#         LinvLs = F.linalg.trsm(L, Ls)
#         Linvmu = F.linalg.trsm(L, mu)
#         LinvSLinvT = F.linalg.syrk(LinvLs)
#         wv = F.linalg.trsm(L, Linvmu, transpose=True)
#
#         Kxt = kern.K(F, Z, X, **kern_params)
#         Ktt_diag = kern.Kdiag(F, X, **kern_params)
#
#         f_mean = F.linalg.gemm2(Kxt, wv, True, False)
#
#         LinvKxt = F.linalg.trsm(L, Kxt)
#         tmp = F.linalg.gemm2(LinvSLinvT, LinvKxt)
#         var = F.expand_dims(Ktt_diag - F.sum(F.square(LinvKxt), axis=-2) + F.sum(tmp*LinvKxt, axis=-2), axis=-1)
#
#         f_samples = F.random.normal(shape=(self.num_samples,) + f_mean.shape[1:], dtype=f_mean.dtype) * F.sqrt(var) + f_mean
#
#         # y_samples = f_samples + F.random.normal(shape=f_samples.shape, dtype=f_samples.dtype) * F.sqrt(noise_var)
#
#         return {self.model.Y.uuid: f_samples, 'F_mean': f_mean, 'F_var':var}


class SVGPRegression(Module):
    """
    Stochastic Variational Gaussian Process Regression module with Gaussian likelihood
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
        super(SVGPRegression, self).__init__(
            inputs=inputs, outputs=None, input_names=input_names,
            output_names=output_names, dtype=dtype, ctx=ctx)
        self.mean_func = mean_func
        self.kernel = kernel

    def _generate_outputs(self, output_shapes=None):
        """
        Generate the output of the module with given output_shapes.
        """
        if output_shapes is None:
            Y_shape = self.X.shape[:-1] + (1,)
        else:
            Y_shape = output_shapes['random_variable']
        self.set_outputs([Variable(shape=Y_shape)])

    def _build_module_graphs(self, output_variables):
        """
        Generate a model graph for GP regression module.
        """
        Y = output_variables['random_variable']
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
            mean=0, variance=graph.noise_var, rand_gen=self._rand_gen,
            dtype=self.dtype, ctx=self.ctx))
        graph.mean_func = self.mean_func
        graph.kernel = graph.U.factor.kernel
        post = Posterior(graph)
        post.qU_cov_diag = Variable(shape=(M,), transformation=PositiveTransformation())
        post.qU_cov_W = Variable(shape=(M, M))
        post.qU_mean = Variable(shape=(M, Y.shape[-1]))
        return graph, [post]

    def _attach_default_inference_algorithms(self):
        observed = [v for k, v in self.inputs] + \
            [v for k, v in self.outputs]
        self.attach_log_pdf_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SVGPRegr_log_pdf(
                self._module_graph, self._extra_graphs[0], observed),
            alg_name='svgp_log_pdf')

        observed = [v for k, v in self.inputs]
        self.attach_draw_samples_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=ForwardSamplingAlgorithm(
                self._module_graph, observed),
            alg_name='svgp_sampling')

    @staticmethod
    def define_variable(X, kernel, noise_var, shape=None, inducing_inputs=None,
                        inducing_num=10, mean_func=None, rand_gen=None,
                        dtype=None, ctx=None):
        """
        Creates and returns a set of random variable drawn from a Gaussian
        process.

        :param X: the input variables on which the random variables are
        conditioned.
        :type X: Variable
        :param kernel: the kernel of Gaussian process
        :type kernel: Kernel
        :param shape: the shape of the random variable(s) (the default shape is
        the same shape as *X* but the last dimension is changed to one.)
        :type shape: tuple or [tuple]
        :param mean_func: the mean function of Gaussian process
        :type mean_func: N/A
        :param rand_gen: the random generator (default: MXNetRandomGenerator)
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context)
        :type ctx: None or mxnet.cpu or mxnet.gpu
        """
        gp = SVGPRegression(
            X=X, kernel=kernel, noise_var=noise_var,
            inducing_inputs=inducing_inputs, inducing_num=inducing_num,
            mean_func=mean_func, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable
