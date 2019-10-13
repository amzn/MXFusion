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
from ...runtime.distributions import PointMassRuntime


class PointMass(UnivariateDistribution):
    """
    The Point Mass distribution.

    :param location: the location of the point mass.
    """
    runtime_dist_class = PointMassRuntime

    def __init__(self, location):
        inputs = [('location', location)]
        input_names = ['location']
        output_names = ['random_variable']
        super(PointMass, self).__init__(inputs=inputs, outputs=None,
                                        input_names=input_names,
                                        output_names=output_names)

    @staticmethod
    def define_variable(location, shape=None):
        """
        Creates and returns a random variable drawn from a Normal distribution.

        :param location: the location of the point mass.
        :param shape: Shape of random variables drawn from the distribution. If non-scalar, each variable is drawn iid.
        :returns: RandomVariable drawn from the distribution specified.
        """

        p = PointMass(location=location)
        p._generate_outputs(shape=shape)
        return p.random_variable
