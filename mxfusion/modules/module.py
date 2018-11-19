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
from mxnet import initializer
from mxnet.gluon import ParameterDict
from ..common.config import get_default_dtype
from ..components.distributions.random_gen import MXNetRandomGenerator
from ..common.exceptions import ModelSpecificationError
from ..components.factor import Factor
from ..components.variables.variable import VariableType
from ..util.inference import realize_shape


class Module(Factor):
    """
    The base class for a probabilistic module.

    A probabilistic module is a combination of model dentition and Inference algorithms.
    It acts as a factor and are defined as such during model definition,
    producing random variables like a plain probabilistic distribution.
    It differs from a plain distribution in that to compute it's log_pdf
    and draw_samples functions, it uses a full Inference method.

    :param inputs: the input variables
    :type inputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
    :param outputs: the output variables
    :type outputs: List of tuples of name to node e.g. [('random_variable': Variable y)] or None
    :param input_names: the names of all the input variables
    :type input_names: [str]
    :param output_names: the names of all the output variables
    :type output_names: [str]
    :param rand_gen: the random generator (default: MXNetRandomGenerator).
    :type rand_gen: RandomGenerator
    :param dtype: the data type for float point numbers.
    :type dtype: numpy.float32 or numpy.float64
    :param ctx: the mxnet context (default: None/current context).
    :type ctx: None or mxnet.cpu or mxnet.gpu
    """

    def __init__(self, inputs, outputs, input_names,
                 output_names, rand_gen=None, dtype=None, ctx=None):
        super(Module, self).__init__(
            inputs=inputs, outputs=outputs, input_names=input_names,
            output_names=output_names)
        self._rand_gen = MXNetRandomGenerator if rand_gen is None else \
            rand_gen
        self.dtype = get_default_dtype() if dtype is None else dtype
        self.ctx = ctx
        self._module_graph = None
        self._extra_graphs = []
        self._log_pdf_algorithms = {}
        self._draw_samples_algorithms = {}
        self._prediction_algorithms = {}

    def __contains__(self, key):
        return any([key in g for g in [self._module_graph] + self._extra_graphs])

    def __getitem__(self, key):
        if key in self._module_graph:
            return self._module_graph[key]
        else:
            for g in self._extra_graphs:
                if key in g:
                    return g[key]
        return self._module_graph[key]
    def _generate_outputs(self, output_shapes):
        """
        Generate the output of the module with given output_shapes.

        :param output_shape: the shapes of all the output variables
        :type output_shape: {str: tuple}
        """
        raise NotImplementedError

    def _build_module_graphs(self):
        """
        The internal method for constructing the internal factor graphs of the module. This method needs to be overridden by specific probabilistic modules.

        :returns: model, extra factor graphs
        :rtypes: Model, [FactorGraph]
        """
        raise NotImplementedError

    def _attach_default_inference_algorithms(self):
        """
        The internal method for attaching default inference algorithms of the module. This method needs to be overridden by specific probabilistic modules.
        """
        raise NotImplementedError

    def set_outputs(self, variables):
        """
        This method overrides the set_outputs method of Factor. It triggers the initialization produces of a probabilistic module including building the factor graphs and attaching default inference algorithms.

        :param variables: The list of variables to be set as the outputs of the module
        :type variable: Variable or (Variable,)
        """
        variables = [variables] if not isinstance(variables, (list, tuple)) \
            else variables
        outputs = {name: variable for name, variable in
                   zip(self.output_names, variables)}
        self.successors = [(k, v) for k, v in outputs.items()]
        self._module_graph, self._extra_graphs = \
            self._build_module_graphs()
        self._attach_default_inference_algorithms()

    def expose_hidden_parameters_as_input(self, name, variable):
        """
        Expose a hidden parameter of the module as an input variable.

        :param name: the name of the resulting input variable
        :type name: str
        :param variable: the reference to the internal variable that will be exposed.
        :type variable: Variable
        """
        v = variable.replicate_self()
        if name in self._inputs_names:
            raise ModelSpecificationError('The module '+str(self)+' already has the input with the name '+name+'.')
        self._inputs_names.append(name)
        self.inputs = self.inputs + [(name, v)]

    @property
    def hidden_parameters(self):
        """
        The UUIDs of all the hidden parameters.
        """
        vars = []
        for g in [self._module_graph]+self._extra_graphs:
            vars.extend(g.get_parameters(
                        excluded=set([v.uuid for k, v in self.inputs]),
                        include_inherited=True))
        return [v.uuid for v in vars]

    def initialize_hidden_parameters(self, param_dict=None, excluded=None,
                                     constants=None):
        """
        Initialize all the hidden parameters.

        :param param_dict: the MXNet ParameterDict for parameter initialization
        :type param_dict: MXNet ParameterDict
        :param excluded: the set of variables that are excluded from initialization
        :type excluded: set(str(UUID))
        :param constants: the constants discovered during initialization, to be used for shape inference
        :type constants: {str(UUID): float or int}
        """
        if param_dict is None:
            param_dict = ParameterDict()
        if excluded is None:
            excluded = set()
        if constants is None:
            constants = {}
        for g in [self._module_graph]+self._extra_graphs:
            for var in g.get_parameters(
                    excluded=set([v.uuid for _, v in self.inputs] +
                                 [v.uuid for _, v in self.outputs]
                                 ).union(constants.keys()).union(excluded),
                    include_inherited=True):

                var_shape = realize_shape(var.shape, constants)
                init = initializer.Constant(var.initial_value_before_transformation) \
                    if var.initial_value is not None else None
                param_dict.get(name=var.uuid, shape=var_shape, dtype=self.dtype,
                               allow_deferred_init=True, init=init)
        return param_dict

    def get_names_from_uuid(self, uuids):
        """
        Get the names of a set of input or output variables given their UUIDs.

        :param uuids: the list of UUIDs
        :type uuids: [str(UUID)]
        """
        uuid_to_names = {v.uuid: k for k, v in self.inputs}
        uuid_to_names.update({v.uuid: k for k, v in self.outputs})
        return tuple(sorted([uuid_to_names[uuid] for uuid in uuids if uuid in
                             uuid_to_names]))

    def attach_log_pdf_algorithms(self, targets, conditionals, algorithm,
                                  alg_name=None):
        """
        Attach an inference algorithm for computing the log_pdf of the module.

        :param targets: Variables to compute the log probability of.
        :type targets: tuple of str
        :param conditionals: variables to condition the probabilities on.
        :type conditionals: tuple of str
        :param algorithm: the inference algorithm to compute log probability of
        the module.
        :type algorithm: InferenceAlgorithm
        """
        self._attach_algorithm(self._log_pdf_algorithms, targets, conditionals, algorithm, alg_name)

    def attach_draw_samples_algorithms(self, targets, conditionals, algorithm,
                                   alg_name=None):
        """
        Attach an inference algorithm for drawing samples from the module.

        :param targets: a list of names of arguments to draw samples from.
        :type targets: tuple of str
        :param conditionals: Variables to condition the samples on.
        :type conditionals: tuple of str
        :param algorithm: the inference algorithm to draw samples of the chosen target variables from the module.
        :type algorithm: InferenceAlgorithm
        """
        self._attach_algorithm(self._draw_samples_algorithms, targets, conditionals, algorithm, alg_name)


    def attach_prediction_algorithms(self, targets, conditionals, algorithm,
                             alg_name=None):
        """
        Attach an inference algorithm for prediction from the module.

        :param targets: a list of names of arguments to predict.
        :type targets: tuple of str
        :param conditionals: Variables to condition the prediction on.
        :type conditionals: tuple of str
        :param algorithm: the inference algorithm to predict the chosen target variables from the module.
        :type algorithm: InferenceAlgorithm
        """
        self._attach_algorithm(self._prediction_algorithms, targets, conditionals, algorithm, alg_name)

    def _attach_algorithm(self, algorithms, targets, conditionals, algorithm, alg_name):
        """
        Attaches the given algorithm to the algorithms data structure based on targets, conditionals, and alg_name.
        Also sets 'm.{alg_name} = algorithm'.
        """
        targets, conditionals = self._preprocess_attach_parameters(targets, conditionals)
        alg_name = self._set_algorithm_name(alg_name, algorithm)
        if conditionals in algorithms:
            return self._attach_duplicate_conditional_algorithm(algorithms, targets, conditionals, algorithm, alg_name)
        else:
            algorithms[conditionals] = [(targets, algorithm, alg_name)]
            return algorithms

    def _preprocess_attach_parameters(self, targets, conditionals):
        """
        Sorts and returns as tuples the targets and conditionals used during attachment.
        """
        if targets is not None:
            targets = tuple(sorted(targets))
        if conditionals is not None:
            conditionals = tuple(sorted(conditionals))
        return targets, conditionals

    def _set_algorithm_name(self, alg_name, algorithm):
        """
        Sets the attribute of self with the algorithm name, overriding an old algorithm that had the same name. If something other than an InferenceAlgorithm has that name, prints a warning and returns None for alg_name.
        """

        from ..inference.inference_alg import InferenceAlgorithm
        if alg_name is not None:
            if not hasattr(self, alg_name):
                setattr(self, alg_name, algorithm)
            elif isinstance(getattr(self, alg_name), InferenceAlgorithm):
                setattr(self, alg_name, algorithm)
            else:
                warnings.warn('Something ({}) in this module ({}) is already using the attribute \"{}\". Skipping setting that name to the algorithm.'.format(str(getattr(self, alg_name)),str(self), str(alg_name)))
                alg_name = None
        return alg_name

    def _attach_duplicate_conditional_algorithm(self, algorithms, targets, conditionals, algorithm, alg_name):
        """
        Mutates the algorithms object, adding the new algorithm to it.
        Also removes the name of an old inference algorithm if it had the same (targets, conditional) pair as the new algorithm.
        """
        methods = algorithms[conditionals]
        no_match = True
        # For each algorithm that already uses those same conditionals
        for i, (i_targets, i_algorithm, i_name) in enumerate(methods):
            # If the targets are also the same, remove the old one
            # because this (targets, conditionals) pair should be unique across algorithms.
            if targets == i_targets:
                # remove the name of the old algorithm
                if i_name is not None and i_name != alg_name:
                    delattr(self, i_name)
                methods[i] = (targets, algorithm, alg_name)
                no_match = False
                break
        if no_match:
            algorithms[conditionals].append(
                (targets, algorithm, alg_name))

    def log_pdf(self, F, variables, targets=None):
        """
        Compute the logarithm of the probability/probability density of a set of random variables in the Module. The set of random
        variables are specified in the "target" argument and any necessary conditional variables are specified in the "conditionals" argument.
        Any relevant constants are specified in the "constants" argument.

        :param F: the MXNet computation mode (``mxnet.symbol`` or ``mxnet.ndarray``).
        :param variables: The set of variables
        :type variables: {UUID : MXNet NDArray or MXNet Symbol}
        :param targets: Variables to compute the log probability of.
        :type targets: {uuid : mxnet NDArray or mxnet Symbol}
        :returns: the sum of the log probability of all the target variables.
        :rtype: mxnet NDArray or mxnet Symbol
        """
        alg = self._get_algorithm_for_target_conditional_pair(self._log_pdf_algorithms, targets, variables, exact_match=True)
        result = alg.compute(F, variables)
        return result

    def draw_samples(self, F, variables, num_samples=1, targets=None):
        """
        Draw samples from the target variables of the Module. If the ``targets`` argument is None, draw samples from all the variables
        that are *not* in the conditional variables. If the ``targets`` argument is given, this method returns a list of samples of variables in the order of the target argument, otherwise it returns a dict of samples where the keys are the UUIDs of variables and the values are the samples.

        :param F: the MXNet computation mode (``mxnet.symbol`` or ``mxnet.ndarray``).
        :param variables: The set of variables
        :type variables: {UUID : MXNet NDArray or MXNet Symbol}
        :param num_samples: The number of samples to draw for the target variables.
        :type num_samples: int
        :param targets: a list of Variables to draw samples from.
        :type targets: [UUID]
        :returns: the samples of the target variables.
        :rtype: (MXNet NDArray or MXNet Symbol,) or {str(UUID): MXNet NDArray or MXNet Symbol}
        """
        alg = self._get_algorithm_for_target_conditional_pair(self._draw_samples_algorithms, targets, variables)
        alg.num_samples = num_samples
        alg.target_variables = targets
        return alg.compute(F, variables)

    def predict(self, F, variables, num_samples=1, targets=None):
        """
        Predict some variables.

        :param F: the MXNet computation mode (``mxnet.symbol`` or ``mxnet.ndarray``).
        :param variables: The set of variables
        :type variables: {UUID : MXNet NDArray or MXNet Symbol}
        :param num_samples: The number of samples to draw for the target variables if sampling is used for prediction. (optional)
        :type num_samples: int
        :param targets: a list of Variables to predict.
        :type targets: [UUID]
        :returns: the sum of the log probability of all the target variables.
        :rtype: mxnet NDArray or mxnet Symbol
        """
        alg = self._get_algorithm_for_target_conditional_pair(self._prediction_algorithms, targets, variables)
        alg.num_samples = num_samples
        alg.target_variables = targets
        return alg.compute(F, variables)

    def _get_algorithm_for_target_conditional_pair(self, algorithms, targets, variables, exact_match=False):
        """
        Searches through the algorithms to find the right algorithm for the target/conditional pair.
        :param exact_match: This indicates whether the targets passed in must be precisely those in the algorithm, or whether a subset of targets will suffice.
        """
        if targets is None:
            target_names = tuple(sorted(self.output_names.copy()))
        else:
            target_names = self.get_names_from_uuid(targets)
        conditionals_names = self.get_names_from_uuid(variables.keys())
        conditionals_names = conditionals_names if not exact_match else tuple(sorted(set(conditionals_names) - set(target_names)))

        if conditionals_names in algorithms:
            algs = algorithms[conditionals_names]
            target_names = set(target_names)
            for t, alg, _ in algs:
                if not exact_match and target_names <= set(t):
                    return alg
                if exact_match and target_names == set(t):
                    return alg

        raise ModelSpecificationError("The targets-conditionals pattern for draw_samples computation "+str((target_names, conditionals_names))+" cannot find a matched inference algorithm.")

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
        for g in [self._module_graph]+self._extra_graphs:
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

    def _clone_algorithms(self, algorithms, replicant):
        """
        Clones all of the algorithms using the replicant graphs.
        """
        algs = {}
        for conditionals, algorithms in algorithms.items():
            for targets, algorithm, alg_name in algorithms:
                graphs_index = {g: i for i,g in enumerate(self._extra_graphs)}
                extra_graphs = [replicant._extra_graphs[graphs_index[graph]] for graph in algorithm.graphs if graph in graphs_index]
                algs[conditionals] = (targets, algorithm.replicate_self(replicant._module_graph, extra_graphs), alg_name)
        return algs

    def reconcile_with_module(self, previous_module):
        from ..models import FactorGraph
        current_graphs = [self._module_graph] + self._extra_graphs
        primary_previous_graph = previous_module._module_graph
        secondary_previous_graphs = previous_module._extra_graphs
        primary_current_graph = self._module_graph
        component_map = FactorGraph.reconcile_graphs(current_graphs, primary_previous_graph, secondary_previous_graphs=secondary_previous_graphs, primary_current_graph=primary_current_graph)
        return component_map

    def replicate_self(self, attribute_map=None):
        """
        The copy constructor for the function.
        """
        replicant = super(Module, self).replicate_self(attribute_map)

        replicant._rand_gen = self._rand_gen
        replicant.dtype = self.dtype
        replicant.ctx = self.ctx
        replicant._module_graph = self._module_graph.clone()

        # Note this assumes the extra graphs are A) posteriors and B) derived from self._module_graph.
        replicant._extra_graphs = [m.clone(self._module_graph) for m in
                             self._extra_graphs]

        replicant._log_pdf_algorithms = self._clone_algorithms(self._log_pdf_algorithms, replicant)
        replicant._draw_samples_algorithms = self._clone_algorithms(self._draw_samples_algorithms, replicant)
        replicant._prediction_algorithms = self._clone_algorithms(self._prediction_algorithms, replicant)
        return replicant

    def load_module(self, module_json):
        from ..models import FactorGraph
        self._module_graph = FactorGraph(module_json['graphs'][0]['name']).load_from_json(module_json['graphs'][0])
        if len(module_json['graphs']) > 1:
            self._extra_graphs = [FactorGraph(extra_graph['name']).load_from_json(extra_graph) for extra_graph in module_json['graphs'][1:]]
        return self


    def as_json(self):
        mod_dict = super(Module, self).as_json()
        graphs = [g.as_json()for g in [self._module_graph] + self._extra_graphs]
        mod_dict['graphs'] = graphs
        return mod_dict
