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


from abc import ABC, abstractmethod
import numpy as np
from ...common.config import get_default_MXNet_mode


class VariableTransformation(ABC):
    """
    Abstract class for transformations and constraints applied to Variables.
    """
    @abstractmethod
    def transform(self, var, F=None, dtype=None):
        """
        Forward transformation.

        :param var: Variable to be transformed.
        :type var: mx.ndarray or mx.sym
        :param F: Mode to run MxNet in.
        :type F: mxnet.ndarray or mxnet.symbol
        :param dtype: data type.
        :type dtype: e.g. np.float32
        """
        pass

    @abstractmethod
    def inverseTransform(self, out_var, F=None, dtype=None):
        """
        Inverse transformation.

        :param out_var: Variable to be transformed.
        :type out_var: mx.ndarray or mx.sym
        :param F: Mode to run MxNet in.
        :type F: mxnet.ndarray or mxnet.symbol
        :param dtype: data type.
        :type dtype: e.g. np.float32
        """
        pass

class Softplus(VariableTransformation):

    """
    Transformation to apply the Softplus and inverse Softplus functions.
    f = log(1+exp(x))+c
    f^-1 = log(exp(x-c)-1)
    """
    def __init__(self, offset):
        self._offset = offset

    def transform(self, var, F=None, dtype=None):
        """
        Forward transformation.

        :param var: Variable to be transformed.
        :type var: mx.ndarray or mx.sym
        :param F: Mode to run MxNet in.
        :type F: mxnet.ndarray or mxnet.symbol
        :param dtype: data type.
        :type dtype: e.g. np.float32
        """
        F = get_default_MXNet_mode() if F is None else F
        return F.Activation(var, act_type='softrelu') + self._offset

    def inverseTransform(self, out_var, F=None, dtype=None):
        """
        Inverse transformation.

        :param out_var: Variable to be transformed.
        :type out_var: mx.ndarray or mx.sym
        :param F: Mode to run MxNet in.
        :type F: mxnet.ndarray or mxnet.symbol
        :param dtype: data type.
        :type dtype: e.g. np.float32
        """
        F = get_default_MXNet_mode() if F is None else F
        # TODO: make a customized operator for this computation to be overflow
        # robust.
        return F.log(F.expm1(out_var - self._offset))


class PositiveTransformation(Softplus):
    """
    Transformation positively constrain a Variable. Wrapper for the Softplus transformation with offset 0.
    """
    def __init__(self):
        """
        Initializes as Softplus transformation with 0 offset.
        """
        super(PositiveTransformation, self).__init__(offset=0.)


class Logistic(VariableTransformation):
    """
    Transformation to constraint a variable to lie between two values.
    """
    def __init__(self, lower, upper):
        """
        :param lower: Lower bound
        :param upper: Upper bound
        """
        if lower >= upper:
            raise ValueError('The lower bound is above the upper bound')
        self._lower, self._upper = lower, upper
        self._difference = self._upper - self._lower

    def transform(self, var, F=None, dtype=None):
        """
        Forward transformation.

        :param var: Variable to be transformed.
        :type var: mx.ndarray or mx.sym
        :param F: Mode to run MxNet in.
        :type F: mxnet.ndarray or mxnet.symbol
        :param dtype: data type.
        :type dtype: e.g. np.float32
        """
        F = get_default_MXNet_mode() if F is None else F
        return self._lower + self._difference * F.Activation(var, act_type='sigmoid')

    def inverseTransform(self, out_var, F=None, dtype=None):
        """
        Inverse transformation.

        :param out_var: Variable to be transformed.
        :type out_var: mx.ndarray or mx.sym
        :param F: Mode to run MxNet in.
        :type F: mxnet.ndarray or mxnet.symbol
        :param dtype: data type.
        :type dtype: e.g. np.float32
        """
        F = get_default_MXNet_mode() if F is None else F
        # Clip out_var to be within bounds to avoid taking log of zero or infinity
        clipped_out_var = F.clip(out_var, self._lower + 1e-10, self._upper - 1e-10)
        return F.log((clipped_out_var - self._lower) / (self._upper - clipped_out_var))
