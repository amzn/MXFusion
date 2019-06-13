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
from ...runtime.distributions import WishartRuntime


class Wishart(Distribution):
    """
    The Wishart distribution.

    :param degrees_of_freedom: Degrees of freedom of the Wishart distribution.
    :type degrees_of_freedom: Variable
    :param scale: Scale matrix of the distribution.
    :type scale: Variable
    """
    runtime_dist_class = WishartRuntime

    def __init__(self, degrees_of_freedom, scale):
        inputs = [('degrees_of_freedom', degrees_of_freedom), ('scale', scale)]
        input_names = ['degrees_of_freedom', 'scale']
        output_names = ['random_variable']
        super(Wishart, self).__init__(inputs=inputs, outputs=None,
                                      input_names=input_names,
                                      output_names=output_names)

    @staticmethod
    def define_variable(shape, degrees_of_freedom, scale):
        """
        Creates and returns a random variable drawn from a Wishart distribution.

        :param degrees_of_freedom: Degrees of freedom of the distribution.
        :param scale: Scale matrix of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :returns: the random variables drawn from the Wishart distribution.
        :rtypes: Variable
        """
        wishart = Wishart(degrees_of_freedom=degrees_of_freedom, scale=scale)
        wishart._generate_outputs(shape=shape)
        return wishart.random_variable
