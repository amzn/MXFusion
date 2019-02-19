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


import warnings
import numpy as np
import mxnet as mx
from mxnet import initializer
from mxnet import ndarray
from mxnet.gluon import ParameterDict
from ..components.variables import VariableType, Variable
from ..components import ModelComponent
from ..util.inference import realize_shape
from ..common.config import get_default_device, get_default_dtype
from ..components.functions.gluon_func_eval import GluonFunctionEvaluation


class InferenceParameters(object):
    """
    The parameters and outcomes of an inference method.

    InferenceParameters is a pool of memory that contains a mapping from uuid to two types of memories
    (MXNet ParameterDict and Constants).

    :param constants: Specify a list of model variables as constants
    :type constants: {ModelComponent.uuid : mxnet.ndarray}
    :param dtype: data type for internal numerical representation
    :type dtype: {numpy.float64, numpy.float32, 'float64', 'float32'}
    :param context: The MXNet context
    :type context: {mxnet.cpu or mxnet.gpu}
    """
    def __init__(self, constants=None, dtype=None, context=None):
        self.dtype = dtype if dtype is not None else get_default_dtype()
        self.mxnet_context = context if context is not None else get_default_device()
        self._constants = {}
        self._var_ties = {}
        if constants is not None:
            constant_uuids = {
                (k.uuid if isinstance(k, ModelComponent) else k): v
                for k, v in constants.items()}
            self._constants.update(constant_uuids)
        self._params = ParameterDict()

    def update_constants(self, constants):
        """
        Update the constants.

        :param constants: The constants to be updated.
        :type constants: {Variable: float or MXNet NDArray}
        """
        self.constants.update({
            (k.uuid if isinstance(k, ModelComponent) else k): v
            for k, v in constants.items()})

    def initialize_params(self, graphs, observed_uuid):
        """
        :param graphs: a list of graphs in which the parameters will be optimized.
        :type graphs: a list of FactorGraph
        :param observed_uuid: Parameter Variables that are passed in directly as data, not to be inferred.
        :type observed_uuid: list, set
        """
        if self._params is not None:
            warnings.warn("InferenceParameters has already been initialized.  The existing one will be overwritten.")

        self._params = ParameterDict()
        for g in graphs:
            # load in parameterdict from external gluon blocks.
            for f in g.functions.values():
                if isinstance(f, GluonFunctionEvaluation):
                    self._params.update(
                        f.function.collect_gluon_parameters())

            for var in g.get_constants():
                self._constants[var.uuid] = var.constant

            excluded = set(self._constants.keys()).union(observed_uuid)
            for var in g.get_parameters(excluded=excluded,
                                        include_inherited=False):
                var_shape = realize_shape(var.shape, self._constants)
                init = initializer.Constant(var.initial_value_before_transformation) \
                    if var.initial_value is not None else None

                self._params.get(name=var.uuid, shape=var_shape,
                                 dtype=self.dtype,
                                 allow_deferred_init=True, init=init)
            for m in g.modules.values():
                m.initialize_hidden_parameters(self._params, excluded, self._constants)

        self._params.initialize(ctx=self.mxnet_context)

    def initialize_with_carryover_params(self, graphs, observed_uuid, var_ties,
                                         carryover_params):
        """
        :param graphs: a list of graphs in which the parameters will be optimized.
        :type graphs: a list of FactorGraph
        :param observed_uuid: Parameter Variables that are passed in directly as data, not to be inferred.
        :type observed_uuid: {UUID : mx.ndarray}
        :param var_ties: A dictionary of variable maps that are tied together and use the MXNet Parameter of the dict
        value's uuid.
        :type var_ties: { UUID to tie from : UUID to tie to }
        :param carryover_params: list of InferenceParameters containing the outcomes of previous inference algorithms.
        :type carryover_params: [InferenceParameters]
        """
        # TODO: var_ties is discarded at the moment.

        var_uuid = set()
        for g in graphs:
            var_uuid = var_uuid.union(set(g.variables.keys()))
            for m in g.modules.values():
                var_uuid = var_uuid.union(set(m.hidden_parameters))

        carryover_pairs = {}
        for carryover in carryover_params:
            for uuid, v in carryover.param_dict.items():
                if uuid in var_uuid:
                    if uuid in carryover_pairs:
                        warnings.warn('The variable with UUID '+uuid+' exists in multiple carryover parameter sets.')
                    carryover_pairs[uuid] = v

        # self._var_ties = var_ties.copy()
        # for g in graphs:
        #     # TODO: check the behavior of var_ties in graph
        #     self._var_ties.update(g.var_ties)
        # for v_uuid in self.constants:
        #     if v_uuid in self._var_ties:
        #         del self._var_ties[v_uuid]

        observed_uuid = set(observed_uuid).union(carryover_pairs.keys())
        self.initialize_params(graphs, observed_uuid)

        # carryover_pairs = {
        #     to_var_uuid: carryover.param_dict[to_var_uuid]
        #     for from_var_uuid, to_var_uuid in self._var_ties.items()
        #     for carryover in carryover_params
        #     if to_var_uuid in carryover.param_dict}
        self._params.update(carryover_pairs)

    @property
    def param_dict(self):
        return self._params

    @property
    def constants(self):
        return self._constants

    @property
    def var_ties(self):
        return self._var_ties

    def __getitem__(self, key, ctx=None):
        if not isinstance(key, Variable):
            raise KeyError("The access key of inference parameter needs to be Variable, but got "+str(type(key))+".")
        pkey = key.inherited_name if key.isInherited else key.uuid
        val = self._params.get(pkey).data(ctx)
        if key.transformation is not None:
            val = key.transformation.transform(val)
        return val

    def __setitem__(self, key, item):
        if not isinstance(key, Variable):
            raise KeyError("The access key of inference parameter needs to be Variable, but get "+str(type(key))+".")

        if key.type == VariableType.PARAMETER:
            if key.transformation is not None:
                item = key.transformation.inverseTransform(item)
            self._params.get(key.uuid).set_data(item)
        elif key.type == VariableType.CONSTANT:
            self._params.get(key.uuid)._value = item

    # Override contains so that it doesn't use the __getitem__ method.
    def __contains__(self, k):
        return k in self.__dict__

    @staticmethod
    def load_parameters(uuid_map=None,
                        mxnet_parameters=None,
                        variable_constants=None,
                        mxnet_constants=None,
                        context=None, dtype=None,
                        current_params=None):
        """
        Loads back a set of InferenceParameters from files.
        :param mxnet_parameters: These are the parameters of
                                     the previous inference algorithm.
        These are in a {uuid: mx.nd.array} mapping.
        :type mxnet_parameters: Dict of {uuid: mx.nd.array}
        :param mxnet_constants: These are the constants in mxnet format
                                    from the previous inference algorithm.
        These are in a {uuid: mx.nd.array} mapping.
        :type mxnet_constants:  Dict of {uuid: mx.nd.array}
        :param variable_constants: These are the constants in
                                       primitive format from the previous
        inference algorithm.
        :type variable_constants: dict of {uuid: constant primitive}
        """
        def with_uuid_map(item, uuid_map):
            if uuid_map is not None:
                return uuid_map[item]
            else:
                return item
        ip = InferenceParameters(context=context, dtype=dtype)

        mapped_params = {with_uuid_map(k, uuid_map): v
                         for k, v in mxnet_parameters.items()}

        new_paramdict = ParameterDict()
        if current_params is not None:
            new_paramdict.update(current_params)

        # Do this because we need to map the uuids to the new Model
        # before loading them into the ParamDict
        for name, mapped_param in mapped_params.items():
            new_paramdict[name]._load_init(mapped_param, ip.mxnet_context)
        ip._params = new_paramdict

        new_mxnet_constants = {}
        new_variable_constants = {}
        new_variable_constants = {with_uuid_map(k, uuid_map): v
                                  for k, v in variable_constants.items()}
        new_mxnet_constants = {with_uuid_map(k, uuid_map): v
                               for k, v in mxnet_constants.items()}

        ip._constants = {}
        ip._constants.update(new_variable_constants)
        ip._constants.update(new_mxnet_constants)
        return ip

    def get_serializable(self):
        """
        Returns three dicts:
         1. MXNet parameters {uuid: mxnet parameters, mx.nd.array}.
         2. MXNet constants {uuid: mxnet parameter (only constant types), mx.nd.array}
         3. Other constants {uuid: primitive numeric types (int, float)}
         :returns: Three dictionaries: MXNet parameters, MXNet constants, and other constants (in that order)
         :rtypes: {uuid: mx.nd.array}, {uuid: mx.nd.array}, {uuid: primitive (int/float)}
        """

        mxnet_parameters = {key: value._reduce() for key, value in self._params.items()}

        mxnet_constants = {uuid: value
                           for uuid, value in self._constants.items()
                           if isinstance(value, mx.ndarray.ndarray.NDArray)}

        variable_constants = {uuid: value
                              for uuid, value in self._constants.items()
                              if uuid not in mxnet_constants}

        return mxnet_parameters, mxnet_constants, variable_constants
