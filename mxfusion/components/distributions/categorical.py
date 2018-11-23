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


from .univariate import UnivariateDistribution
from ...common.config import get_default_MXNet_mode


class Categorical(UnivariateDistribution):
    """
    The Categorical distribution.

    :param log_prob: the logarithm of the probability being in each of the classes.
    :type log_prob: Variable
    :param num_classes: the number of classes.
    :type num_classes: int
    :param one_hot_encoding: If true, the random variable is one-hot encoded.
    :type one_hot_encoding: boolean
    :param normalization: If true, a softmax normalization is applied.
    :type normalization: boolean
    :param axis: the axis in which the categorical distribution is assumed (default: -1).
    :type axis: int
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """
    def __init__(self, log_prob, num_classes, one_hot_encoding=False,
                 normalization=True, axis=-1, rand_gen=None, dtype=None,
                 ctx=None):
        inputs = [('log_prob', log_prob)]
        input_names = ['log_prob']
        output_names = ['random_variable']
        super(Categorical, self).__init__(
            inputs=inputs, outputs=None,
            input_names=input_names,
            output_names=output_names,
            rand_gen=rand_gen, dtype=dtype,
            ctx=ctx)
        if axis != -1:
            raise NotImplementedError("The Categorical distribution currently only supports the last dimension to be the class label dimension, i.e., axis == -1.")
        self.axis = axis
        self.normalization = normalization
        self.one_hot_encoding = one_hot_encoding
        self.num_classes = num_classes

    def replicate_self(self, attribute_map=None):
        """
        This functions as a copy constructor for the object.
        In order to do a copy constructor we first call ``__new__`` on the class which creates a blank object.
        We then initialize that object using the methods standard init procedures, and do any extra copying of attributes.

        Replicates this Factor, using new inputs, outputs, and a new uuid.
        Used during model replication to functionally replicate a factor into a new graph.

        :param inputs: new input variables of the factor.
        :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        :param outputs: new output variables of the factor.
        :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
        """
        replicant = super(Categorical, self).replicate_self(attribute_map=attribute_map)
        replicant.axis = self.axis
        replicant.normalization = self.normalization
        replicant.one_hot_encoding = self.one_hot_encoding
        replicant.num_classes = self.num_classes
        return replicant

    def log_pdf_impl(self, log_prob, random_variable, F=None):
        """
        Computes the logarithm of probabilistic mass function of the Categorical distribution.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param log_prob: the logarithm of the probability being in each of the classes.
        :type log_prob: MXNet NDArray or MXNet Symbol
        :param random_variable: the point to compute the log pdf for.
        :type random_variable: MXNet NDArray or MXNet Symbol
        :returns: log pdf of the distribution.
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F

        if self.normalization:
            log_prob = F.log_softmax(log_prob, axis=self.axis)

        if self.one_hot_encoding:
            logL = F.sum(F.broadcast_mul(random_variable, log_prob),
                         axis=self.axis) * self.log_pdf_scaling
        else:
            logL = F.pick(log_prob, index=random_variable, axis=self.axis)
            logL = logL * self.log_pdf_scaling
        return logL

    def draw_samples_impl(self, log_prob, rv_shape, num_samples=1, F=None):
        """
        Draw a number of samples from the Categorical distribution.

        :param log_prob: the logarithm of the probability being in each of the classes.
        :type log_prob: MXNet NDArray or MXNet Symbol
        :param rv_shape: the shape of each sample.
        :type rv_shape: tuple
        :param num_samples: the number of drawn samples (default: one).
        :int num_samples: int
        :param F: the MXNet computation mode (mxnet.symbol or mxnet.ndarray).
        :returns: a set samples of the Categorical distribution
        :rtypes: MXNet NDArray or MXNet Symbol
        """
        F = get_default_MXNet_mode() if F is None else F
        rv_ndim = len(rv_shape)

        if self.normalization:
            log_prob = F.log_softmax(log_prob, axis=self.axis)

        log_prob = F.transpose(log_prob, axes=list(range(rv_ndim))+[rv_ndim])
        if num_samples != log_prob.shape[0]:
            log_prob = F.broadcast_to(log_prob, (num_samples,)+rv_shape[:-1]+(self.num_classes,))
        samples = self._rand_gen.sample_multinomial(log_prob)
        if self.one_hot_encoding:
            samples = F.one_hot(samples, depth=self.num_classes)
        samples = F.reshape(samples, shape=(num_samples,) + rv_shape)
        return samples

    @staticmethod
    def define_variable(log_prob, num_classes, shape=None, one_hot_encoding=False, normalization=True, axis=-1,
                        rand_gen=None, dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Categorical distribution.

        :param log_prob: the logarithm of the probability being in each of the classes.
        :type log_prob: Variable
        :param num_classes: the number of classes.
        :type num_classes: int
        :param shape: the shape of the Categorical variable.
        :type shape: tuple of int
        :param one_hot_encoding: If true, the random variable is one-hot encoded.
        :type one_hot_encoding: boolean
        :param normalization: If true, a softmax normalization is applied.
        :type normalization: boolean
        :param axis: the axis in which the categorical distribution is assumed (default: -1).
        :type axis: int
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu
        :returns: RandomVariable drawn from the Categorical distribution.
        :rtypes: Variable
        """
        cat = Categorical(
            log_prob=log_prob, num_classes=num_classes,
            one_hot_encoding=one_hot_encoding, normalization=normalization,
            axis=axis, rand_gen=rand_gen, dtype=dtype, ctx=ctx)
        cat._generate_outputs(shape=shape)
        return cat.random_variable
