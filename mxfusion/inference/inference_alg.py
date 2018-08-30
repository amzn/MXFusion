from abc import ABC, abstractmethod
from mxnet.gluon import HybridBlock
from ..components.variables import VariableType
from ..components.variables import add_sample_dimension_arrays
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
        super(ObjectiveBlock, self).__init__(prefix=prefix, params=params)
        self._infr_method = infr_method
        self._constants = constants
        self._data_def = data_def
        self._var_trans = var_trans
        self._var_ties = var_ties
        for name in params:
            if name not in excluded:
                setattr(self, name, params.get(name))

    def hybrid_forward(self, F, x, *args, **kw):

        for to_uuid, from_uuid in self._var_ties.items():
            kw[to_uuid] = kw[from_uuid]
        data = {k: v for k, v in zip(self._data_def, args)}
        data = add_sample_dimension_arrays(F, data)
        for k, v in self._var_trans.items():
            kw[k] = v.transform(kw[k], F=F)
        kw = add_sample_dimension_arrays(F, kw)
        constants = add_sample_dimension_arrays(F, self._constants)
        obj = self._infr_method.compute(F=F, data=data, parameters=kw,
                                        constants=constants)
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

    def __init__(self, model, observed, extra_graphs=None):
        self._model_graph = model
        self._extra_graphs = extra_graphs if extra_graphs is not None else []
        self._graphs = [model] if extra_graphs is None else \
            [model] + extra_graphs
        self._observed = observed
        self._observed_uuid = variables_to_UUID(observed)
        self._observed_names = [v.name for v in observed]

    @property
    def observed_variables(self):
        return self._observed

    @property
    def observed_variable_UUIDs(self):
        return self._observed_uuid

    @property
    def observed_variable_names(self):
        return self._observed_names

    @property
    def model(self):
        """
        return the model definition.
        """
        return self._model_graph

    @property
    def graphs(self):
        return self._graphs

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
        :param rv_scaling: the scaling factor of random variables
        :type rv_scaling: {UUID: scaling factor}
        :returns: the Gluon block computing the outcome of inference
        :rtype: mxnet.gluon.HybridBlock
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
        block = ObjectiveBlock(infr_method=self, params=params.param_dict,
                               constants=params.constants,
                               data_def=data_def, var_trans=var_trans,
                               var_ties=var_ties, excluded=excluded)
        return block

    @abstractmethod
    def compute(self, F, data, parameters, constants):
        """
        The abstract method for the computation of the inference algorithm

        :param F: the execution context (mxnet.ndarray or mxnet.symbol)
        :type F: Python module
        :param data: the data variables for inference
        :type data: {Variable: mxnet.ndarray.ndarray.NDArray or
            mxnet.symbol.symbol.Symbol}
        :param parameters: the parameters for inference
        :type parameters: {Variable: mxnet.ndarray.ndarray.NDArray or
            mxnet.symbol.symbol.Symbol}
        :param constants: the constants for inference
        :type parameters: {Variable: mxnet.ndarray.ndarray.NDArray or
            mxnet.symbol.symbol.Symbol}
        :returns: the outcome of the inference algorithm
        :rtype: mxnet.ndarray.ndarray.NDArray or mxnet.symbol.symbol.Symbol
        """
        raise NotImplementedError
