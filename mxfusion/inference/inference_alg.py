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
from mxnet.gluon import HybridBlock
from mxnet import autograd
from ..common.constants import SET_PARAMETER_PREFIX
from ..components.variables import VariableType
from ..components.variables import add_sample_dimension_to_arrays
from ..util.inference import variables_to_UUID


class ObjectiveBlock(HybridBlock):
    """
    Wrapping the MXNet codes with a MXNet Gluon HybridBlock for execution.

    :param infr_method: the pointer to a function that contains the actual MXNet codes.
    :type infr_method: a pointer to a function
    :param constants: the variables with constant values
    :type constants: {Variable UUID: int or float or mxnet.ndarray}
    :param data_def: a list of variable UUID, which corresponds to the order of variables expected as the positional arguments in "hybrid_forward".
    :type data_def: [UUID]
    :param var_trans: the transformations applied variables
    :type var_trains: {UUID: VariableTransformation}
    :param var_ties: A dictionary of variables that are tied and use the MXNet Parameter of the dict value uuid.
    :type var_ties: { UUID of source variable: UUID of target variable}
    :param excluded: a set of variables excluded from being set as Block parameters.
    :type excluded: set(UUID) or [UUID]
    :param prefix: the prefix of this Gluon block
    :type prefix: str
    :param params: the ParameterDict of this Gluon block
    :type params: mxnet.gluon.ParameterDict
    """
    def __init__(self, infr_method, constants, data_def, var_trans, var_ties,
                 excluded, prefix='', params=None):
        super(ObjectiveBlock, self).__init__(prefix=prefix, params=params.param_dict)
        self._infr_method = infr_method
        self._constants = constants
        self._data_def = data_def
        self._var_trans = var_trans
        self._var_ties = var_ties
        self._infr_params = params
        for name in params.param_dict:
            if name not in excluded:
                setattr(self, name, params.param_dict.get(name))

    def hybrid_forward(self, F, x, *args, **kw):
        """
        This function does all the preprocesses and postprocesses for the execution of a InferenceAlgorithm.

        :param F: the MXNet computation mode
        :type F: mxnet.symbol or mxnet.ndarray
        :param x: a dummy variable to enable the execution of this Gluon block
        :type x: MXNet NDArray or MXNet Symbol
        :param *arg: all the positional arguments, which correspond to the data provided to the InferenceAlgorithm.
        :type *arg: list of MXNet NDArray or MXNet Symbol
        :param **kw: all the keyword arguments, which correspond to the parameters that may require gradients.
        :type kw: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the InferenceAlgorithm that are determined by the inference algorithm.
        :rtypes: {str: MXNet NDArray or MXNet Symbol}
        """
        for to_uuid, from_uuid in self._var_ties.items():
            kw[to_uuid] = kw[from_uuid]
        data = {k: v for k, v in zip(self._data_def, args)}
        variables = add_sample_dimension_to_arrays(F, data)
        for k, v in self._var_trans.items():
            kw[k] = v.transform(kw[k], F=F)
        add_sample_dimension_to_arrays(F, kw, out=variables)
        add_sample_dimension_to_arrays(F, self._constants, out=variables)
        obj = self._infr_method.compute(F=F, variables=variables)
        with autograd.pause():
            # An inference algorithm may directly set the value of a parameter instead of computing its gradient.
            # This part handles the setting of parameters.
            for k, v in variables.items():
                if k.startswith(SET_PARAMETER_PREFIX):
                    self._infr_params[v[0]] = v[1]
        return obj


class InferenceAlgorithm(ABC):
    """
    The abstract class for an inference algorithm. A concrete inference
    algorithm will inherit this class and overload the "compute" function with
    the actual computation logic.

    :param model: the definition of the probabilistic model
    :type model: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param extra_graphs: a list of extra FactorGraph used in the inference
                         algorithm.
    :type extra_graphs: [FactorGraph]
    """

    def replicate_self(self, model, extra_graphs=None):

        replicant = self.__class__.__new__(self.__class__)
        replicant._model_graph = model
        replicant._extra_graphs = extra_graphs if extra_graphs is not None else []
        observed = [replicant.model[o] for o in self._observed_uuid]
        replicant._observed = set(observed)
        replicant._observed_uuid = variables_to_UUID(observed)
        replicant._observed_names = [v.name for v in observed]
        return replicant


    def __init__(self, model, observed, extra_graphs=None):
        self._model_graph = model
        self._extra_graphs = extra_graphs if extra_graphs is not None else []
        self._graphs = [model] if extra_graphs is None else \
            [model] + extra_graphs
        self._observed = set(observed)
        self._observed_uuid = variables_to_UUID(observed)
        self._observed_names = [v.name for v in observed]

    @property
    def observed_variables(self):
        """
        The observed variables in this inference algorithm.
        """
        return self._observed

    @property
    def observed_variable_UUIDs(self):
        """
        The UUIDs of the observed variables in this inference algorithm.
        """
        return self._observed_uuid

    @property
    def observed_variable_names(self):
        """
        The names (if exist) of the observed variables in this inference algorithm.
        """
        return self._observed_names

    @property
    def model(self):
        """
        the model that the inference algorithm applies to
        """
        return self._model_graph

    @property
    def graphs(self):
        """
        all the factor graphs that the inference algorithm uses
        """
        return self._graphs

    def prepare_executor(self, rv_scaling=None):
        """
        Prepare the creation of an executor. This includes collecting the list of variable transformations and the list of the variables that are inherited from external Gluon blocks, and setting log_pdf_scaling for random variables.

        :param rv_scaling: The scaling of log_pdf of the random variables that are set by users for data sub-sampling or mini-batch learning.
        :type rv_scaling: {UUID: float}
        :returns: the list of the variable transformations and the list of the variables that are excluded from being set as Gluon block parameters (see the excluded argument of __init__ of ObjectiveBlock).
        :rtypes: {str(UUID): Transformation}, set(str(UUID))
        """
        excluded = set()
        var_trans = {}
        rv_scaling = {} if rv_scaling is None else rv_scaling
        for g in self._graphs:
            for v in g.variables.values():
                if v.type == VariableType.PARAMETER and v.transformation is not None:
                    var_trans[v.uuid] = v.transformation
                if v.type == VariableType.PARAMETER and v.isInherited:
                    excluded.add(v.uuid)
                if v.type == VariableType.RANDVAR:
                    if v.uuid in rv_scaling:
                        v.factor.log_pdf_scaling = rv_scaling[v.uuid]
                    else:
                        v.factor.log_pdf_scaling = 1
        return var_trans, excluded

    def create_executor(self, data_def, params, var_ties, rv_scaling=None):
        """
        Create a MXNet Gluon block to carry out the computation.

        :param data_def: a list of unique ID of data variables. The order of
            variables in the list corresponds to the order of variable in the
            positional arguments when calling the Gluon Block.
        :type data_def: [UUID of Variable (str)]
        :param params: The parameters of model involved in inference.
        :type params: InferenceParameters
        :param var_ties: A dictionary of variables that are tied and use the MXNet Parameter of the dict value uuid.
        :type var_ties: { UUID of source variable: UUID of target variable}
        :param rv_scaling: The scaling of log_pdf of the random variables that are set by users for data sub-sampling or mini-batch learning.
        :type rv_scaling: {UUID: float}
        :returns: the Gluon block computing the outcome of inference
        :rtype: mxnet.gluon.HybridBlock
        """
        var_trans, excluded = self.prepare_executor(rv_scaling=rv_scaling)
        for m in self.model.modules.values():
            var_trans_m, excluded_m = m.prepare_executor(rv_scaling=rv_scaling)
            var_trans.update(var_trans_m)
            excluded = excluded.union(excluded_m)
        block = ObjectiveBlock(infr_method=self, params=params,
                               constants=params.constants,
                               data_def=data_def, var_trans=var_trans,
                               var_ties=var_ties, excluded=excluded)
        return block

    @abstractmethod
    def compute(self, F, variables):
        """
        The abstract method for the computation of the inference algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """
        raise NotImplementedError

    def set_parameter(self, variables, target_variable, target_value):
        """
        Set the value of a variable as the artifacts of this inference algorithm. This triggers to set the value to the corresponding variable into InferenceParameters at the end of inference.

        :param variables: the set of MXNet arrays that holds the values of
        all the variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :param target_variable: the variable that a value is set to
        :type target_variable: Variable
        :param target_value: the value to be set
        :type target_value: MXNet NDArray or float
        """
        variables[target_variable.uuid] = target_value
        variables[SET_PARAMETER_PREFIX+target_variable.uuid] = \
            (target_variable, target_value)


class SamplingAlgorithm(InferenceAlgorithm):
    """
    The base class of sampling algorithms.

    :param model: the definition of the probabilistic model
    :type model: Model
    :param observed: A list of observed variables
    :type observed: [Variable]
    :param num_samples: the number of samples used in estimating the variational lower bound
    :type num_samples: int
    :param target_variables: (optional) the target variables to sample
    :type target_variables: [UUID]
    :param extra_graphs: a list of extra FactorGraph used in the inference
                         algorithm.
    :type extra_graphs: [FactorGraph]
    """

    def __init__(self, model, observed, num_samples=1, target_variables=None,
                 extra_graphs=None):
        super(SamplingAlgorithm, self).__init__(
            model=model, observed=observed, extra_graphs=extra_graphs)
        self.num_samples = num_samples
        self.target_variables = target_variables

    def compute(self, F, variables):
        """
        The abstract method for the computation of the inference algorithm.

        If inference algorithm is used for gradient based optimizations, it should return two values. The first for the loss function, the second the gradient of the loss function.

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param variables: the set of MXNet arrays that holds the values of
        variables at runtime.
        :type variables: {str(UUID): MXNet NDArray or MXNet Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol. If gradient based, will return two values. The first the loss function, the second the gradient of the loss function.
        """
        raise NotImplementedError
