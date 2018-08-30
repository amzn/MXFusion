from abc import ABC, abstractmethod
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
