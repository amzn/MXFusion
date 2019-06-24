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
from ...runtime.distributions import LaplaceRuntime


class Laplace(UnivariateDistribution):
    """
    The one-dimensional Laplace distribution. The Laplace distribution can be defined over a scalar random variable
    or an array of random variables. In case of an array of random variables, the location and scale are broadcasted
    to the shape of the output random variable (array).

    :param location: Location of the Laplace distribution.
    :type location: Variable
    :param scale: Scale of the Laplace distribution.
    :type scale: Variable
    """
    runtime_dist_class = LaplaceRuntime

    def __init__(self, location, scale):
        inputs = [('location', location), ('scale', scale)]
        input_names = [k for k, _ in inputs]
        output_names = ['random_variable']
        super(Laplace, self).__init__(inputs=inputs, outputs=None,
                                      input_names=input_names,
                                      output_names=output_names)

    @staticmethod
    def define_variable(location=0., scale=1., shape=None):
        """
        Creates and returns a random variable drawn from a Laplace distribution.

        :param location: Location of the distribution.
        :param scale: Scale of the distribution.
        :param shape: the shape of the random variable(s).
        :type shape: tuple or [tuple]
        :returns: the random variables drawn from the Laplace distribution.
        :rtypes: Variable
        """
        var = Laplace(location=location, scale=scale)
        var._generate_outputs(shape=shape)
        return var.random_variable
