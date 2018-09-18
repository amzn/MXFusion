import numpy as np
from ..module import Module
from ...models import Model
from ...components.variables.variable import Variable
from ...components.distributions import GaussianProcess, Normal
from ...inference.inference_alg import InferenceAlgorithm


class GPRegr_log_pdf(InferenceAlgorithm):
    def __init__(self, model, observed):
        super(GPRegr_log_pdf, self).__init__(model=model, observed=observed)

    def compute(self, F, variables):
        X = variables[self.model.X]
        Y = variables[self.model.Y]
        noise_var = variables[self.model.noise_var]
        D = Y.shape[-1]
        N = X.shape[-2]
        kern = self.model.kernel
        kern_params = kern.fetch_parameters(variables)

        K = kern.K(F, X, **kern_params) + F.eye(N, dtype=X.dtype) * noise_var
        L = F.linalg.potrf(K)

        if self.model.mean_func is not None:
            mean = self.model.mean_func(F, X)
            Y = Y - mean
        LinvY = F.linalg.trsm(L, Y)
        logdet_l = F.linalg.sumlogdiag(F.abs(L))

        return - logdet_l * D - F.sum(F.square(LinvY) + np.log(2. * np.pi)) / 2


# class GPPrediction(InferenceAlgorithm):
#     def __init__(self, model, observed):
#         super(GPPrediction, self).__init__(model=model, observed=observed)
#
#     def compute(self, F, data, parameters, constants):
#         X = data[self.model.X]
#         X_cond = data[self.model.X_cond]
#         Y_cond = data[self.model.Y_cond]
#         kern = self.model.kernel
#         kern_params = kern.fetch_parameters(parameters)
#
#         Kxt = kern.K(F, X_cond, X, **kern_params)
#         Ktt = kern.Kdiag(F, X, **kern_params)
#         Kxx = kern.K(F, X_cond, **kern_params)
#         L = F.linalg.potrf(Kxx)
#         LInvY = F.linalg.trsm(L, Y_cond)
#         LinvKxt = F.linalg.trsm(L, Kxt)
#
#         mu = F.linalg.gemm2(LinvKxt, LInvY, True, False)
#         var = Ktt - F.sum(F.square(LinvKxt), axis=1)
#         return mu, var


class GPRegression(Module):
    """
    """

    def __init__(self, X, kernel, noise_var, mean_func=None, rand_gen=None,
                 dtype=None, ctx=None):
        if not isinstance(X, Variable):
            X = Variable(value=X)
        if not isinstance(noise_var, Variable):
            noise_var = Variable(value=noise_var)
        inputs = [('X', X), ('noise_var', noise_var)]
        # + [(k, v) for k, v in kernel.parameters.items()]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(GPRegression, self).__init__(
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
        graph = Model(name='gp_regression')
        graph.X = self.X.replicate_self()
        graph.noise_var = self.noise_var.replicate_self()
        graph.F = GaussianProcess.define_variable(
            X=graph.X, kernel=self.kernel, shape=Y.shape,
            mean_func=self.mean_func, rand_gen=self.rand_gen, dtype=self.dtype,
            ctx=self.ctx)
        graph.Y = Y.replicate_self()
        graph.Y.set_prior(Normal(
            mean=graph.F, variance=graph.noise_var, rand_gen=self.rand_gen,
            dtype=self.dtype, ctx=self.ctx))
        graph.mean_func = self.mean_func
        graph.kernel = graph.F.factor.kernel
        return graph, []

    def _attach_default_inference_algorithms(self):
        observed = [v for k, v in self.inputs] + \
            [v for k, v in self.outputs]
        self.attach_log_prob_algorithms(
            targets=self.output_names, conditionals=self.input_names,
            algorithm=GPRegr_log_pdf(self._module_graph, observed))

    @staticmethod
    def define_variable(X, kernel, noise_var, shape=None, mean_func=None,
                        rand_gen=None, dtype=None, ctx=None):
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
        gp = GPRegression(
            X=X, kernel=kernel, noise_var=noise_var, mean_func=mean_func,
            rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        gp._generate_outputs({'random_variable': shape})
        return gp.random_variable
