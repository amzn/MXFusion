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


from ..factor import Factor
from .random_gen import MXNetRandomGenerator
from ...util.inference import realize_shape, \
    broadcast_samples_dict
from ...common.config import get_default_dtype


class Distribution(Factor):
    """
    The base class of a probability distribution associated with one or a set of random variables.

    :param inputs: the input variables that parameterize the probability distribution.
    :type inputs: {name: Variable}
    :param outputs: the random variables drawn from the distribution.
    :type outputs: {name: Variable}
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, inputs, outputs, input_names, output_names, rand_gen=None, dtype=None, ctx=None):
        super(Distribution, self).__init__(inputs=inputs, outputs=outputs,
                                           input_names=input_names,
                                           output_names=output_names)
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        self.dtype = get_default_dtype() if dtype is None else dtype
        self.ctx = ctx
        self.log_pdf_scaling = 1

    def replicate_self(self, attribute_map=None):
        replicant = super(Distribution, self).replicate_self(attribute_map)
        replicant._rand_gen = self._rand_gen
        replicant.dtype = self.dtype
        replicant.ctx = self.ctx
        replicant.log_pdf_scaling = 1
        return replicant

    def log_pdf(self, F, variables, targets=None):
        """
        Computes the logarithm of the probability density/mass function (PDF/PMF) of the distribution. 
        The inputs and outputs variables are fetched from the *variables* argument according to their UUIDs.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        kwargs = {}
        for name, var in self.inputs:
            kwargs[name] = variables[var.uuid]
        for name, var in self.outputs:
            kwargs[name] = variables[var.uuid]
        kwargs = broadcast_samples_dict(F, kwargs)
        return self.log_pdf_impl(F=F, **kwargs)

    def log_pdf_impl(self, F, **kwargs):
        """
        The implementation of log_pdf for a specific distribution.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log pdf of the distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        raise NotImplementedError

    def log_cdf(self, F=None, **kwargs):
        """
        Computes the logarithm of the cumulative distribution function (CDF) of the distribution.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: log cdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        raise NotImplementedError

    def draw_samples(self, F, variables, num_samples=1, targets=None,
                     always_return_tuple=False):
        """
        Draw a number of samples from the distribution. All the dependent variables are automatically collected from a
        dictionary of variables according to the UUIDs of the dependent variables.

        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :param variables: the set of variables where the dependent variables are collected from.
        :type variables: {str(UUID): MXNet NDArray or Symbol}
        :param num_samples: the number of drawn samples (default: one).
        :type num_samples: int
        :param always_return_tuple: return the samples in a tuple of shape one. This allows easy programming when there
        are potentially multiple output variables.
        :type always_return_tuple: boolean
        :returns: a set samples of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
        """
        kwargs = {}
        for name, var in self.inputs:
            kwargs[name] = variables[var.uuid]
        kwargs = broadcast_samples_dict(F, kwargs, num_samples=num_samples)
        kwargs['rv_shape'] = realize_shape(self.outputs[0][1].shape, variables)
        s = self.draw_samples_impl(F=F, num_samples=num_samples, **kwargs)
        if always_return_tuple and not isinstance(s, (tuple, list)):
            s = (s,)
        return s

    def draw_samples_impl(self, rv_shape, num_samples=1, F=None, **kwargs):
        """
        The implementation of draw_samples for a specific distribution.

        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple, [tuple]
        :param num_samples: the number of drawn samples (default: one).
        :type num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol or [MXNet NDArray or MXNet Symbol]
        """
        raise NotImplementedError

    @staticmethod
    def define_variable(shape=None, rand_gen=None, dtype=None, ctx=None, **kwargs):
        """
        Define a random variable that follows from the specified distribution.

        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :param kwargs: the input variables that parameterize the probability
        distribution.
        :type kwargs: {name: Variable}
        :returns: the random variables drawn from the distribution.
        :rtypes: Variable or [Variable]
        """
        raise NotImplementedError
