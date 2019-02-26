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


from ..variables import Variable
from .univariate import UnivariateDistribution


class PointMass(UnivariateDistribution):
    """
    The Point Mass distribution.

    :param location: the location of the point mass.
    """
    def __init__(self, location, rand_gen=None, dtype=None, ctx=None):
        inputs = [('location', location)]
        input_names = ['location']
        output_names = ['random_variable']
        super(PointMass, self).__init__(inputs=inputs, outputs=None,
                                        input_names=input_names,
                                        output_names=output_names,
                                        rand_gen=rand_gen, dtype=dtype, ctx=ctx)

    def log_pdf_impl(self, location, random_variable, F=None):
        """
        Computes the logarithm of probabilistic density function of the normal distribution.

        :param F: MXNet computation type <mx.sym, mx.nd>.
        :param location: the location of the point mass.
        :param random_variable: the point to compute the logpdf for.
        :returns: An operator chain to compute the logpdf of the Normal distribution.
        """
        return 0.

    def draw_samples_impl(self, location, rv_shape, num_samples=1, F=None):
        if num_samples == location.shape[0]:
            return location
        else:
            return F.broadcast_to(
                location, shape=(num_samples,)+location.shape[1:])

    @staticmethod
    def define_variable(location, shape=None, rand_gen=None, dtype=None, ctx=None):
        """
        Creates and returns a random variable drawn from a Normal distribution.

        :param location: the location of the point mass.
        :param shape: Shape of random variables drawn from the distribution. If non-scalar, each variable is drawn iid.
        :param rand_gen: the random generator (default: MXNetRandomGenerator).
        :type rand_gen: RandomGenerator
        :param dtype: the data type for float point numbers.
        :type dtype: numpy.float32 or numpy.float64
        :param ctx: the mxnet context (default: None/current context).
        :type ctx: None or mxnet.cpu or mxnet.gpu

        :returns: RandomVariable drawn from the distribution specified.
        """
        if not isinstance(location, Variable):
            location = Variable(value=location)

        p = PointMass(location=location, rand_gen=rand_gen, dtype=dtype,
                      ctx=ctx)
        p._generate_outputs(shape=shape)
        return p.random_variable
