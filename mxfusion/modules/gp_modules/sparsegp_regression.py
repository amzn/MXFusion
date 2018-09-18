import numpy as np
from mxnet import autograd
from ..module import Module
from ...models import Model, Posterior
from ...components.variables.variable import Variable
from ...components.distributions import GaussianProcess, Normal, ConditionalGaussianProcess, MultivariateNormal
from ...inference.variational import VariationalInference


class SparseGPRegr_log_pdf(VariationalInference):
    def __init__(self, model, posterior, observed, jitter=0.):
        super(SparseGPRegr_log_pdf, self).__init__(
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

        Kuu = kern.K(F, Z, **kern_params)
        if self.jitter > 0.:
            Kuu = Kuu + F.eye(M, dtype=Z.dtype) * self.jitter
        Kuf = kern.K(F, Z, X, **kern_params)
        Kff_diag = kern.Kdiag(F, X, **kern_params)

        L = F.linalg.potrf(Kuu)
        LinvKuf = F.linalg.trsm(L, Kuf)

        A = F.eye(M, dtype=Z.dtype) + \
            F.broadcast_div(F.linalg.syrk(LinvKuf), noise_var)
        LA = F.linalg.potrf(A)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            Y = Y - mean
        LAInvLinvKufY = F.linalg.trsm(LA, F.linalg.gemm2(LinvKuf, Y))

        logL = - D*F.linalg.sumlogdiag(LA)
        logL = logL - F.sum(F.sum(F.square(Y)/noise_var + (np.log(2. * np.pi) +
                            F.log(noise_var)), axis=-1), axis=-1)/2
        logL = logL + F.sum(F.sum(
            F.square(LAInvLinvKufY)/(2*F.square(noise_var)), axis=-1), axis=-1)
        logL = logL - D*F.sum(Kff_diag, axis=-1)/(2*noise_var)
        logL = logL + F.sum(F.sum(F.square(LinvKuf)/(2.*noise_var),
                            axis=-1), axis=-1)

        with autograd.pause():
            wv = variables[self.graphs[1].wv]
            wv[:] = F.broadcast_div(F.linalg.trsm(L, F.linalg.trsm(LA, LAInvLinvKufY, transpose=True), transpose=True), noise_var)[0]

        return logL


class SparseGPRegression(Module):
    """
    """

    def __init__(self, X, kernel, noise_var, inducing_inputs=None,
                 inducing_num=10, mean_func=None,
                 rand_gen=None, dtype=None, ctx=None):
        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)
        if inducing_inputs is None:
            inducing_inputs = Variable(shape=(inducing_num, X.shape[-1]))
        inputs = [('X', X), ('inducing_inputs', inducing_inputs),
                  ('noise_var', noise_var)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(SparseGPRegression, self).__init__(
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
            mean_func=self.mean_func, rand_gen=self.rand_gen, dtype=self.dtype,
            ctx=self.ctx)
        # graph.F = ConditionalGaussianProcess.define_variable(
        #     X=graph.X, X_cond=graph.inducing_inputs, Y_cond=graph.U,
        #     kernel=self.kernel, shape=Y.shape, mean_func=self.mean_func,
        #     rand_gen=self.rand_gen, dtype=self.dtype, ctx=self.ctx)
        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(
            mean=0, variance=graph.noise_var, rand_gen=self.rand_gen,
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
        observed = [v for k, v in self.inputs] + \
            [v for k, v in self.outputs]
        self.attach_log_prob_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=SparseGPRegr_log_pdf(self._module_graph, self._extra_graphs[0], observed))

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
        gp = SparseGPRegression(
            X=X, kernel=kernel, noise_var=noise_var,
            inducing_inputs=inducing_inputs, inducing_num=inducing_num,
            mean_func=mean_func, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable
