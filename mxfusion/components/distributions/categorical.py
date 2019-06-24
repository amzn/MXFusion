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
from .univariate import UnivariateDistribution
from ...util.inference import broadcast_samples_dict
from ...runtime.distributions import CategoricalRuntime


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
    """
    runtime_dist_class = CategoricalRuntime

    def __init__(self, log_prob, num_classes, one_hot_encoding=False,
                 normalization=True, axis=-1):
        inputs = [('log_prob', log_prob)]
        input_names = ['log_prob']
        output_names = ['random_variable']
        super(Categorical, self).__init__(
            inputs=inputs, outputs=None,
            input_names=input_names,
            output_names=output_names)
        if axis != -1:
            raise NotImplementedError("The Categorical distribution currently only supports the last dimension to be "
                                      "the class label dimension, i.e., axis == -1.")
        self.axis = axis
        self.normalization = normalization
        self.one_hot_encoding = one_hot_encoding
        self.num_classes = num_classes

    def get_runtime_distribution(self, variables):
        kwargs = self.fetch_runtime_inputs(variables)
        kwargs = broadcast_samples_dict(mx.nd, kwargs)
        return self.runtime_dist_class(axis=self.axis, normalization=self.normalization,
                                       one_hot_encoding=self.one_hot_encoding, num_classes=self.num_classes, **kwargs)

    def replicate_self(self, attribute_map=None):
        """
        This functions as a copy constructor for the object.
        In order to do a copy constructor we first call ``__new__`` on the class which creates a blank object.
        We then initialize that object using the methods standard init procedures, and do any extra copying of
        attributes.

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

    @staticmethod
    def define_variable(log_prob, num_classes, shape=None, one_hot_encoding=False, normalization=True, axis=-1):
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
        :returns: RandomVariable drawn from the Categorical distribution.
        :rtypes: Variable
        """
        cat = Categorical(
            log_prob=log_prob, num_classes=num_classes,
            one_hot_encoding=one_hot_encoding, normalization=normalization,
            axis=axis)
        cat._generate_outputs(shape=shape)
        return cat.random_variable
