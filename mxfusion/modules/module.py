from mxnet.gluon import ParameterDict
from mxnet import initializer
from ..components.variables.variable import VariableType
from ..components.factor import Factor
from ..common.exceptions import ModelSpecificationError
from ..util.inference import realize_shape
from ..common.config import get_default_dtype


class Module(Factor):
    """
    Modules are a combined Model/Posterior + Inference algorithm object in MXFusion. They act as Factors and are defined as such during model definition, producing Random Variables like a typical Distribution. During inference, instead of the Module having a closed form solution, the outside inference algorithm will call the module's internal inference algorithm to perform modular inference.
    """

    def __init__(self, inputs, outputs, input_names,
                 output_names, dtype=None, ctx=None):
        super(Module, self).__init__(
            inputs=inputs, outputs=outputs, input_names=input_names,
            output_names=output_names)
        self.dtype = get_default_dtype() if dtype is None else dtype
        self.ctx = ctx
        self._module_graph = None
        self._extra_graphs = []
        self._log_prob_methods = {}
        self._draw_samples_methods = {}

    def _generate_outputs(self, output_shapes):
        """
        Generate the output of the module with given output_shapes.
        """
        raise NotImplementedError

    def _build_model_graph(self, output_variables):
        """
        Generate a model graph for the module.
        """
        raise NotImplementedError

    def _attach_default_inference_algorithms(self):
        raise NotImplementedError

    def set_outputs(self, variables):
        """
        """
        variables = [variables] if not isinstance(variables, (list, tuple)) \
            else variables
        outputs = {name: variable for name, variable in
                   zip(self.output_names, variables)}
        self._module_graph, self._extra_graphs = \
            self._build_module_graphs(outputs)
        self._attach_default_inference_algorithms()
        self.successors = [(k, v) for k, v in outputs.items()]

    def expose_hidden_parameters_as_input(self, name, variable):
        v = variable.replicate_self()
        if name in self._inputs_names:
            raise ModelSpecificationError('The module '+str(self)+' already has the input with the name '+name+'.')
        self._inputs_names.append(name)
        self.inputs = self.inputs + [(name, v)]

    @property
    def hidden_parameters(self):
        vars = self._module_graph.get_parameters(
            excluded=[v.uuid for k, v in self.inputs], include_inherited=True)
        return [v.uuid for v in vars]

    def initialize_hidden_parameters(self, param_dict=None, constants=None):
        if param_dict is None:
            param_dict = ParameterDict()
        if constants is None:
            constants = {}
        for g in [self._module_graph]+self._extra_graphs:
            for var in g.get_parameters(
                    excluded=set([v.uuid for k, v in self.inputs]).union(constants.keys()),
                    include_inherited=True):

                var_shape = realize_shape(var.shape, constants)
                init = initializer.Constant(var.initial_value_before_transformation) \
                    if var.initial_value is not None else None
                param_dict.get(name=var.uuid, shape=var_shape, dtype=self.dtype,
                               allow_deferred_init=True, init=init)
        return param_dict

    def get_names_from_uuid(self, uuids):
        uuid_to_names = {v.uuid: k for k, v in self.inputs}
        uuid_to_names.update({v.uuid: k for k, v in self.outputs})
        return tuple(sorted([uuid_to_names[uuid] for uuid in uuids if uuid in
                             uuid_to_names]))

    def attach_log_prob_algorithms(self, targets, conditionals, algorithm):
        """
        :param targets: Variables to compute the log probability of.
        :type targets: tuple of str
        :param conditionals: variables to condition the probabilities on.
        :type conditionals: tuple of str
        :param algorithm: the inference algorithm to compute log probability of
        the module.
        :type algorithm: InferenceAlgorithm
        """
        if targets is not None:
            targets = tuple(sorted(targets))
        if conditionals is not None:
            conditionals = tuple(sorted(conditionals))
        self._log_prob_methods[(targets, conditionals)] = algorithm

    def attach_draw_samples_algorithms(self, targets, conditionals, algorithm):
        """
        :param targets: a list of names of arguments to draw samples from.
        :type targets: tuple of str
        :param conditionals: Variables to condition the samples on.
        :type conditionals: tuple of str
        :param algorithm: the inference algorithm to draw samples of the chosen target variables from the module.
        :type algorithm: InferenceAlgorithm
        """
        if targets is not None:
            targets = tuple(sorted(targets))
        if conditionals is not None:
            conditionals = tuple(sorted(conditionals))
        if conditionals in self._draw_samples_methods:
            self._draw_samples_methods[conditionals].append((targets,
                                                             algorithm))
        else:
            self._draw_samples_methods[conditionals] = [(targets, algorithm)]

    def compute_log_prob(self, F, variables, targets=None):
        if targets is None:
            target_names = self.get_names_from_uuid(targets)
        else:
            target_names = tuple(sorted(self.output_names.copy()))
        conditionals_names = self.get_names_from_uuid(variables.keys())
        conditionals_names = tuple(sorted(set(conditionals_names) - set(target_names)))

        if (target_names, conditionals_names) in self._log_prob_methods:
            alg = self._log_prob_methods[(target_names, conditionals_names)]
        else:
            raise ModelSpecificationError("The targets, conditionals pattern for log_prob computation "+str((target_names, conditionals_names))+" cannot find a matched inference algorithm.")
        return alg.compute(F, variables)

    def draw_samples(self, F, variables, num_samples=1, targets=None):
        if targets is None:
            target_names = self.get_names_from_uuid(targets)
        else:
            target_names = tuple(sorted(self.output_names.copy()))
        conditionals_names = self.get_names_from_uuid(variables.keys())

        if conditionals_names in self._draw_samples_methods:
            algs = self._draw_samples_methods[conditionals_names]
            target_names = set(target_names)
            for t, alg in algs:
                if target_names <= t:
                    alg.num_samples = num_samples
                    alg.target_variables = targets
                    return alg.compute(F, variables)
        raise ModelSpecificationError("The targets-conditionals pattern for draw_samples computation "+str((target_names, conditionals_names))+" cannot find a matched inference algorithm.")

    def prepare_executor(self, rv_scaling=None):
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
