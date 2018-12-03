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


from enum import Enum
import mxnet as mx
import numpy as np
from ...common.exceptions import ModelSpecificationError
from ..model_component import ModelComponent
from ...common.config import get_default_dtype


class VariableType(Enum):

    """
    Variables types:

    * CONSTANT: a variable with a constant value
    * PARAMETER: a variable with no prior distribution
    * RANDVAR: a variable that follows a distribution or module
    * FUNCVAR: a variable that is an output of a function
    """

    CONSTANT = 0
    PARAMETER = 1
    RANDVAR = 2
    FUNCVAR = 3


class Variable(ModelComponent):
    """
    Define a variable in MXFusion. The variable can be either a constant, a parameter, an outcome of a function evaluation or a random variable
    following a probabilistic distribution.

    :param value: The value of variable. If it is a numpy or MXNet array, the variable is considered as a constant. If it is a function evaluation, the
        variable is considered as the outcome of a function evaluation. If it is a probabilistic distribution, the variable is considered as a random
        variable. If it is None, the variable is considered as a parameter.
    :type value: (optional) None or numpy array or MXNet array or float or int or FunctionEvaluation or Distribution.
    :param shape: The expected shape of the Variable.
    :type shape: (optional) tuple or None
    :param transformation: The constraint on the domain of the variable, e.g., a positive value.
    :type transformation: (optional) transformation or None
    :param isInherited: True if the Variable is constructed from a GluonBlock parameter.
    :type isInherited: bool
    """
    def __init__(self, value=None, shape=None, transformation=None, isInherited=False, initial_value=None):
        super(Variable, self).__init__()
        self.shape = shape  # For constants, if shape is None then it is inferred from the value
        if self.shape is not None:
            assert isinstance(self.shape, tuple), "Shape is expected to be a tuple or None"
            self.attributes = [s for s in self.shape if isinstance(s, Variable)]
        else:
            self.attributes = []
        # whether the variable is inherited from a Gluon block.
        self.isInherited = isInherited
        self._transformation = transformation
        self._value = None
        if isinstance(initial_value, (int, float)):
            initial_value = mx.nd.array([initial_value],
                                        dtype=get_default_dtype())
        self._initial_value = initial_value
        self.isConstant = False
        from ..distributions import Distribution
        from ...modules.module import Module
        from ..functions.function_evaluation import FunctionEvaluation
        if isinstance(value, (Distribution, Module)):
            self._initialize_as_randvar(value, self.shape, transformation)
        elif isinstance(value, FunctionEvaluation):
            self._initialize_as_funcvar(value, self.shape, transformation)
        else:
            self._initialize_as_param(value, self.shape, transformation)

    @property
    def type(self):
        from ..distributions import Distribution
        from ..functions import FunctionEvaluation
        from ...modules.module import Module
        if self.factor is None:
            if self.isConstant:
                return VariableType.CONSTANT
            else:
                return VariableType.PARAMETER
        elif isinstance(self.factor, (Distribution, Module)):
            return VariableType.RANDVAR
        elif isinstance(self.factor, FunctionEvaluation):
            return VariableType.FUNCVAR

    def as_json(self):
        object_dict = super(Variable, self).as_json()
        object_dict['inherited_name'] = self.inherited_name if self.isInherited else None
        return object_dict

    def replicate_self(self, attribute_map=None):
        """
        This functions as a copy constructor for the object. In order to do a copy constructor we first call ``__new__`` on the class which creates a blank object.
        We then initialize that object using the methods standard init procedures, and do any extra copying of attributes.

        Replicates this Factor, using new inputs, outputs, and a new uuid. Used during model replication to functionally replicate a factor into a new graph.

        :param attribute_map: A mapping from attributes of this object that were Variables to their replicants.
        :type attribute_map: {Variable: replicated Variable}
        """
        if attribute_map is not None:
            shape = tuple(attribute_map[s] if isinstance(s, Variable) else s for s in self.shape)
        else:
            shape = self.shape
        self.attributes = [s for s in shape if isinstance(s, Variable)]
        if self.type == VariableType.RANDVAR or self.type == \
           VariableType.FUNCVAR:
            v = Variable(value=None, shape=shape,
                         transformation=self.transformation)
        elif self.type == VariableType.PARAMETER:
            v = Variable(value=None, shape=shape,
                         transformation=self.transformation)
        elif self.type == VariableType.CONSTANT:
            v = Variable(value=self._value, shape=shape,
                         transformation=self.transformation)
        if self.isInherited:
            v.isInherited = self.isInherited
            v.inherited_name = self.inherited_name
        v._uuid = self.uuid
        v._initial_value = self._initial_value
        return v

    @property
    def shape_str(self):
        ss = '('+', '.join([(str(i.type)[13:] if i.name is None else i.name) if isinstance(i, Variable) else str(i) for i in self.shape])+')'
        return ss

    def display_str(self, temp_name=None):
        if temp_name is not None:
            return temp_name
        name = "Variable"
        if self.name is not None:
            name += " {}".format(self.name)
        name += " ({})".format(self.uuid[:5])
        return name

    def __repr__(self):
        return self.display_str()

    def _initialize_as_param(self, value, shape, transformation):

        if value is None:
            # Initialize as VariableType.PARAMETER
            if shape is None:
                shape = (1,)
        else:
            # Initialize as VariableType.CONSTANT
            self.isConstant = True
            if isinstance(value, np.ndarray):
                if shape is None:
                    shape = value.shape
                if shape != value.shape:
                    raise ModelSpecificationError("Shape mismatch in Variable creation. The numpy array shape " + str(value.shape) + " does not match with the shape argument " + str(shape) + ".")
                value = mx.nd.array(value, dtype=get_default_dtype())
            elif isinstance(value, mx.nd.NDArray):
                if shape is None:
                    shape = value.shape
                if shape != value.shape:
                    raise ModelSpecificationError("Shape mismatch in Variable creation. The MXNet array shape " + str(value.shape) + " does not match with the shape argument " + str(shape) + ".")
            elif isinstance(value, (float, int)):
                shape = (1,)
            else:
                raise ModelSpecificationError("Variable type {} not supported".format(type(value)))
            self._value = value
        self.shape = shape  # Update self.shape with the latest shape

    def _initialize_as_randvar(self, value, shape, transformation):
        if transformation is not None:
            raise NotImplementedError('Constraints on random variables are not supported!')

    def _initialize_as_funcvar(self, value, shape, transformation):
        self._inputs = [value]
        if shape is None:
            raise ModelSpecificationError("The shape argument was not given when defining a variable as the outcome of a function evaluation.")
        if transformation is not None:
            raise NotImplementedError('Constraints on function outputs are not supported!')

    def set_prior(self, distribution):
        """
        Set the prior over this variable.

        :param: distribution The distribution the variable is drawn from.
        """

        self.assign_factor(distribution)

    def assign_factor(self, factor):
        """
        Assign (or reassign) the distribution this variable is drawn from.

        :param: distribution The distribution this variable is drawn from.
        """
        factor.set_outputs(self)

    @property
    def factor(self):
        if self.predecessors is not None and len(self.predecessors) > 0:
            return self.predecessors[0][1]
        else:
            return None

    @property
    def constant(self):
        """
        Return the stored constant.

        :returns: the constant value
        """
        if self.type == VariableType.CONSTANT:
            return self._value
        else:
            raise ModelSpecificationError("The constant property is not accessible for variable with the type "+str(self.type)+".")

    @property
    def transformation(self):
        return self._transformation

    @property
    def initial_value(self):
        return self._initial_value

    @property
    def initial_value_before_transformation(self):
        """
        The initial value of a variable before applying the chosen
        transformation (if exists)
        """
        if self._transformation is None:
            return self._initial_value
        else:
            return self._transformation.inverseTransform(self._initial_value,
                                                         F=mx.nd)

    def __add__(self, y):
        from ..functions.operators import add
        return add(x=self, y=y)

    def __sub__(self, y):
        from ..functions.operators import subtract
        return subtract(self, y)

    def __mul__(self, y):
        from ..functions.operators import multiply
        return multiply(self, y)

    def __truediv__(self, y):
        from ..functions.operators import divide
        return divide(self, y)

    def __pow__(self, y):
        from ..functions.operators import power
        return power(self, y)

    @property
    def T(self):
        from ..functions.operators import transpose
        return transpose(self)
