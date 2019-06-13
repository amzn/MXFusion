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


from .distribution import Distribution
from ...runtime.distributions import MultivariateNormalRuntime


class MultivariateNormal(Distribution):
    """
    The multi-dimensional normal distribution.

    :param mean: Mean of the normal distribution.
    :type mean: Variable
    :param covariance: Covariance matrix of the distribution.
    :type covariance: Variable
    """
    runtime_dist_class = MultivariateNormalRuntime

    def __init__(self, mean, covariance):
        inputs = [('mean', mean), ('covariance', covariance)]
        input_names = ['mean', 'covariance']
        output_names = ['random_variable']
        super(MultivariateNormal, self).__init__(inputs=inputs, outputs=None,
                                     input_names=input_names,
                                     output_names=output_names)

    @staticmethod
    def define_variable(shape, mean=0., covariance=None):
        """
        Creates and returns a random variable drawn from a normal distribution.

        :param mean: Mean of the distribution.
        :param covariance: Variance of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :returns: the random variables drawn from the normal distribution.
        :rtypes: Variable
        """
        normal = MultivariateNormal(mean=mean, covariance=covariance)
        normal._generate_outputs(shape=shape)
        return normal.random_variable
