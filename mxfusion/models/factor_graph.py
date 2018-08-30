from future.utils import raise_from
from uuid import uuid4
import warnings
import networkx as nx
from ..components import Distribution, Factor, ModelComponent, Module, Variable, VariableType
from ..common.exceptions import ModelSpecificationError, InferenceError
from ..components.functions import FunctionEvaluation
from ..components.variables.runtime_variable import expectation


class FactorGraph(object):
    """
    A graph defining how Factor objects relate to one another.

    The two primary functionalities of this class are:

     * ``compute_log_prob`` which computes the log probability of some variables in the graph and
     * ``draw_samples`` which draws samples for some target variables from the graph
    """

    def __init__(self, name, verbose=False):
        """
        Initializes the FactorGraph with a UUID and structures to hold components.
        """
        self.name = name
        self._uuid = str(uuid4())
        self._var_ties = {}

        self._components_graph = nx.DiGraph()
        self._verbose = verbose

    def __repr__(self):
        """
        Return a string summary of this object
        """
        out_str = ''
        for f in self.ordered_factors:
            if isinstance(f, FunctionEvaluation):
                out_str += ', '.join([str(v) for _, v in f.outputs])+' = '+str(f)+'\n'
            elif isinstance(f, Distribution):
                out_str += ', '.join([str(v) for _, v in f.outputs])+' ~ '+str(f)+'\n'
        return out_str[:-1]

    def __getitem__(self, key):
        return self.components[key]

    def __contains__(self, key):
        return key in self.components

    def __setattr__(self, name, value):
        """
        Called whenever an attribute is attached to a FactorGraph object.
        This method adds the attribute's value to it's internal graph representation
        if it's an object derived from the ModelComponent class.

        :param name:  The attribute name.
        :type name: str
        :param value: The value being assigned to the attribute.
        :type value: Anything, but if it is type ModelComponent it is added to some internal data structures.
        """
        if isinstance(value, ModelComponent):
            if self._verbose:
                print("Variable {} ({}) {} ({})".format(name, value.uuid, value.type, value.shape))
            if value.name is not None:
                warnings.warn("The value {} has already been assigned in the model.".format(str(value)))
            value.name = name
            value.graph = self.components_graph
        super(FactorGraph, self).__setattr__(name, value)

    @property
    def components_graph(self):
        """
        Return the Graph object of the component

        :returns: dict of ModelComponents
        :rtype: { UUID : ModelComponent }
        """

        return self._components_graph

    @property
    def components(self):
        """
        Return all the ModelComponents held in the FactorGraph.

        :returns: dict of ModelComponents
        :rtype: { UUID : ModelComponent }
        """

        return {node.uuid: node for node in self.components_graph.nodes()}

    @property
    def distributions(self):
        """
        Return the distributions held in the FactorGraph.

        :returns: dict of Distributions
        :rtype: { UUID : Distribution }
        """

        return {node.uuid: node for node in self.components_graph.nodes() if isinstance(node, Distribution)}

    @property
    def functions(self):
        """
        Return the functions held in the FactorGraph.

        :returns: dict of Functions
        :rtype: { UUID : Function }
        """

        return {node.uuid: node for node in self.components_graph.nodes() if isinstance(node, FunctionEvaluation)}

    @property
    def modules(self):
        """
        Return the modules held in the FactorGraph.

        :returns: dict of Modules
        :rtype: { UUID : Module }
        """

        return {node.uuid: node for node in self.components_graph.nodes() if isinstance(node, Module)}

    @property
    def variables(self):
        """
        Return the variables held in the FactorGraph.

        :returns: dict of Variables
        :rtype: { UUID : Variable }
        """

        return {node.uuid: node for node in self.components_graph.nodes() if isinstance(node, Variable)}

    @property
    def ordered_factors(self):
        """
        Return a sorted list of Factors in the graph.

        :rtype: A topologically sorted list of Factors in the graph.
        """
        return [node for node in nx.topological_sort(self.components_graph)
                                 if isinstance(node, Factor)]

    @property
    def roots(self):
        """
        Return all root notes in the graph.
        """
        return [node for node, degree in self.components_graph.in_degree() if degree == 0]

    @property
    def leaves(self):
        """
        Return all leaf nodes in the graph.
        """
        return [node for node, degree in self.components_graph.out_degree() if degree == 0]

    @property
    def var_ties(self):
        """
        Return a mapping of Variables in the FactorGraph that are tied together (i.e. the same / aliases of each other.)

        :returns: dict of UUIDs of the variables to tie
        :rtype: { uuid_of_var_to_tie : uuid_of_var_to_tie_to }
        """

        return self._var_ties

    def compute_log_prob(self, F, targets, conditionals=None, constants=None):
        """
        Compute the logarithm of the probability/probability density of a set of random variables in the factor graph. The set of random
        variables are specified in the "target" argument and any necessary conditional variables are specified in the "conditionals" argument.
        Any relevant constants are specified in the "constants" argument.

        :param F: the MXNet computation mode (``mxnet.symbol`` or ``mxnet.ndarray``).
        :param targets: Variables to compute the log probability of.
        :type targets: {uuid : RTVariable}
        :param conditionals: variables to condition the probabilities on.
        :type conditionals: {uuid : RTVariable}
        :param constants: the constants that may be used in computation.
        :type constants: {UUID: float or mxnet NDArray or mxnet Symbol}
        :returns: the sum of the log probability of all the target variables.
        :rtype: RTVariable
        """
        constants = {} if constants is None else constants

        variables = targets.copy()
        variables.update(constants)
        if conditionals is not None:
            variables.update(conditionals)

        logL = 0.
        for f in self.ordered_factors:
            if isinstance(f, FunctionEvaluation):
                outcome = f.eval(F=F, variables=variables,
                                 always_return_tuple=True)
                outcome_uuid = [v.uuid for _, v in f.outputs]
                for v, uuid in zip(outcome, outcome_uuid):
                    if uuid in variables:
                        warnings.warn('Function evaluation in FactorGraph.compute_log_prob_RT: the outcome variable '+str(uuid)+' of the function evaluation '+str(f)+' has already existed in the variable set.')
                    variables[uuid] = v
            elif isinstance(f, Distribution):
                if f.random_variable.uuid in targets:
                    logL = logL + F.sum(expectation(F, f.log_pdf(
                        F=F, variables=variables)))
            elif isinstance(f, Module):
                raise NotImplementedError("Modules aren't implemented yet!")
            else:
                raise ModelSpecificationError("There is an object in the factor graph that isn't a factor." + "That shouldn't happen.")
        return logL

    def draw_samples(self, F, num_samples=1, targets=None, conditionals=None,
                     constants=None):
        """
        Draw samples from the target variables of the Factor Graph. If the ``targets`` argument is None, draw samples from all the variables
        that are *not* in the ``conditionals`` argument.

        :param F: the MXNet computation mode (``mxnet.symbol`` or ``mxnet.ndarray``).
        :param num_samples: The number of samples to draw for the target variables.
        :type num_samples: int
        :param targets: a list of Variables to draw samples from.
        :type targets: [UUID]
        :param conditionals: Variables to condition the samples on.
        :type conditionals: {UUID : RTVariable}
        :param constants: the constants that may be used in computation.
        :type constants: {UUID: float or mxnet NDArray or mxnet Symbol}
        :returns: the samples of the target variables.
        :rtype: {UUID : RTVariable}
        """
        constants = {} if constants is None else constants
        variables = {}
        variables.update(constants)
        if conditionals is not None:
            variables.update(conditionals)

        samples = {}
        for f in self.ordered_factors:
            if isinstance(f, FunctionEvaluation):
                outcome = f.eval(F=F, variables=variables,
                                 always_return_tuple=True)
                outcome_uuid = [v.uuid for _, v in f.outputs]
                for v, uuid in zip(outcome, outcome_uuid):
                    if uuid in variables:
                        warnings.warn('Function evaluation in FactorGraph.draw_samples_RT: the outcome of the function evaluation '+str(f)+' has already existed in the variable set.')
                    variables[uuid] = v
                    samples[uuid] = v
            elif isinstance(f, Distribution):
                known = [v in variables for _, v in f.outputs]
                if all(known):
                    continue
                elif any(known):
                    raise InferenceError("Part of the outputs of the distribution " + f.__class__.__name__ + " has been observed!")
                outcome = f.draw_samples(
                    F=F, num_samples=num_samples, variables=variables,
                    always_return_tuple=True)
                outcome_uuid = [v.uuid for _, v in f.outputs]
                for v, uuid in zip(outcome, outcome_uuid):
                    variables[uuid] = v
                    samples[uuid] = v
            elif isinstance(f, Module):
                raise NotImplementedError("Modules aren't implemented yet!")
            else:
                raise ModelSpecificationError("There is an object in the factor graph that isn't a factor." + "That shouldn't happen.")
        if targets:
            samples = {uuid: samples[uuid] for uuid in targets}
        return samples

    def remove_component(self, component):
        """
        Removes the specified component from the factor graph.

        :param component: The component to remove.
        :type component: ModelComponent
        """
        if not isinstance(component, ModelComponent):
                raise ModelSpecificationError(
                    "Attempted to remove an object that isn't a ModelComponent.")

        try:
            self.components_graph.remove_node(component)  # implicitly removes edges
        except Exception as e:
            raise_from(ModelSpecificationError("Attempted to remove a node that isn't in the graph"), e)

        if component.name is not None:
            try:
                self.__delattr__(component.name)
            except Exception as e:
                pass

        component.graph = None

    def _replicate_class(self, **kwargs):
        """
        Returns a new instance of the derived FactorGraph's class.
        """
        return FactorGraph(**kwargs)

    def remove_subgraph(self, node):
        """
        Removes a node and its parent graph recursively.
        """
        if isinstance(node, Variable):
            self.remove_component(node)
            if node.factor is not None:
                self.remove_subgraph(node.factor)
        elif isinstance(node, Factor):
            self.remove_component(node)
            for _, v in node.inputs:
                self.remove_subgraph(v)
        del node

    def replace_subgraph(self, target_variable, new_subgraph):
        """
        Replaces the target_variable with the new_subgraph.

        TODO If the factor of target_variable or new_subgraph as multiple outputs, this will fail.

        :param target_variable: The variable that will be replaced by the new_subgraph.
        :type target_variable: Variable
        :param new_subgraph: We assume this is a Variable right now (aka the graph ends in a Variable).
        :type new_subgraph: Variable
        """
        new_factor = new_subgraph.factor
        new_factor.successors = []

        old_predecessors = target_variable.predecessors
        target_variable.predecessors = []
        for _, p in old_predecessors:
            self.remove_subgraph(p)

        target_variable.assign_factor(new_factor)

    def extract_distribution_of(self, variable):
        """
        Extracts the distribution of the variable passed in, returning a replicated copy of the passed in variable with only its parent
        subgraph attached (also replicated).

        :param variable: The variable to extract the distribution from.
        :type variable: Variable
        :returns: a replicated copy of the passed in variable.
        :rtype: Variable
        """
        def extract_distribution_function(component):
            if isinstance(component, Factor):
                predecessor_direction = 'recursive'
                successor_direction = 'one_level'
                return predecessor_direction, successor_direction
            else:
                predecessor_direction = 'recursive'
                successor_direction = None
                return predecessor_direction, successor_direction
        return variable.replicate(replication_function=extract_distribution_function)

    def clone(self, leaves=None):
        """
        Clones a model, maintaining the same functionality and topology. Replicates all of its ModelComponents with new UUIDs.
        Starts upward from the leaves and copies everything in the graph recursively.

        :param leaves: If None, use the leaves in this model, otherwise use the provided leaves.
        :returns: A tuple of (cloned_model, variable_map)
        """

        new_model = self._replicate_class(name=self.name, verbose=self._verbose)
        var_map = {}  # from old model to new model

        leaves = self.leaves if leaves is None else leaves
        for v in leaves:
            if v.name is not None:
                new_leaf = v.replicate(var_map=var_map,
                           replication_function=lambda x: ('recursive', 'recursive'))
                setattr(new_model, v.name, new_leaf)
            else:
                v.graph = new_model.graph
        for v in new_model.variables.values():
            if v.name is not None:
                setattr(new_model, v.name, v)
        return new_model, var_map

    def get_parameters(self, excluded=None, include_inherited=False):
        """
        Get all the parameters not in the excluded list.

        :param excluded: a list of variables to be excluded.
        :type excluded: set(UUID) or [UUID]
        :param include_inherited: whether inherited variables are included.
        :type include_inherited: boolean
        :returns: the list of contant variables.
        :rtype: [Variable]
        """
        if include_inherited:
            return [v for v in self.variables.values() if (v.type == VariableType.PARAMETER and v.uuid not in excluded)]
        else:
            return [v for v in self.variables.values() if (v.type == VariableType.PARAMETER and v.uuid not in excluded and not v.isInherited)]

    def get_constants(self):
        """
        Get all the contants in the factor graph.

        :returns: the list of constant variables.
        :rtype: [Variable]
        """
        return [v for v in self.variables.values() if v.type == VariableType.CONSTANT]

    @staticmethod
    def reconcile_graphs(current_graphs, primary_previous_graph, secondary_previous_graphs=None, primary_current_graph=None):
        """
        Reconciles two sets of graphs, matching the model components in the previous graph to the current graph.
        This is primarily used when loading back a graph from a file and matching it to an existing in-memory graph in order to load the previous
        graph's paramters correctly.

        :param current_graphs: A list of the graphs we are reconciling a loaded factor graph against. This must be a fully built set of graphs
            generated through the model definition process.
        :param primary_previous_graph: A graph which may have been loaded in from a file and be partially specified, or could be a full graph
            built through model definition.
        :param secondary_previous_graphs: A list of secondary graphs (e.g. posteriors) that share components with the primary_previous_graph.
        :param primary_current_graph: Optional parameter to specify the primary_current_graph, otherwise it is taken to be the model in the
            current_graphs (which should be unique).

        :rtype: {previous ModelComponent : current ModelComponent}
        """
        from .model import Model
        component_map = {}
        current_level = {}
        current_graph = primary_current_graph if primary_current_graph is not None else [graph for graph in current_graphs if isinstance(graph, Model)][0]
        secondary_current_graphs = [graph for graph in current_graphs
                                    if not isinstance(graph, Model)]

        # Map over the named components.
        for c in primary_previous_graph.components.values():
            if c.name:
                current_c = getattr(current_graph, c.name)
                component_map[c.uuid] = current_c.uuid
                current_level[c.uuid] = current_c.uuid

        # Reconcile the primary graph
        FactorGraph._reconcile_graph(current_level, component_map,
                                     current_graph, primary_previous_graph)
        # Reconcile the other graphs
        if not(secondary_current_graphs is None or
               secondary_previous_graphs is None):
            for cg, pg in zip(secondary_current_graphs,
                              secondary_previous_graphs):
                current_level = {pc: cc for pc, cc in component_map.items()
                                 if pc in pg.components.keys()}
                FactorGraph._reconcile_graph(
                    current_level, component_map, cg, pg)

        # Resolve the remaining ambiguities here.
        # if len(component_map) < set([graph.components for graph in previous_graphs])): # TODO the components of all the graphs not just the primary
        #     pass
        return component_map

    @staticmethod
    def _reconcile_graph(current_level, component_map, current_graph, previous_graph):
        """
        Traverses the components in current_level of the current_graph/previous_graph, matching components where possible and generating
        new calls to _reconcile_graph where the graph is still incompletely traversed. This method makes no attempt to resolve ambiguities
        in naming between the graphs and request the user to more completely specify names in their graph if such an ambiguity exists. Such
        naming can be [more] completely specified by attaching names to each leaf node in the original graph.

        :param current_level: A list of items to traverse the graph upwards from.
        :type current_level: [previous ModelComponents]
        :param component_map: The current mapping from the previous graph's MCs to the current_graph's MCs. This is used and modified during reconciliation.
        :type component_map: {previous_graph ModelComponent : current_graph ModelComponent}
        :param current_graph: The current graph to match components against.
        :type current_graph: FactorGraph
        :param previous_graph: The previous graph to match components from.
        :type previous_graph: FactorGraph
        """

        def reconcile_direction(direction, c, current_c, new_level, component_map):
            if direction == 'predecessor':
                previous_neighbors = c.predecessors
                current_neighbors = current_c.predecessors
            elif direction == 'successor':
                previous_neighbors = c.successors
                current_neighbors = current_c.successors
            names = list(map(lambda x: x[0], previous_neighbors))
            duplicate_names = set([x for x in names if names.count(x) > 1])
            for edge_name, node in previous_neighbors:
                if node.uuid not in component_map:
                    if edge_name in duplicate_names:
                        # TODO if all the other parts of the ambiguity are resolved, we have the answer still. Otherwise throw an exception
                        raise Exception("Multiple edges connecting unnamed nodes have the same name, this isn't supported currently.") # TODO Support the ambiguities :)
                    current_node = [item for name, item in current_neighbors if edge_name == name][0]
                    component_map[node.uuid] = current_node.uuid
                    new_level[node.uuid] = current_node.uuid

        new_level = {}
        for c, current_c in current_level.items():
            reconcile_direction('predecessor', previous_graph[c], current_graph[current_c], new_level, component_map)
            """
            TODO Reconciling in both directions currently breaks the reconciliation process and can cause multiple previous_uuid's to map to the same current_uuid. It's unclear why that happens.
            This shouldn't be necessary until we implement multi-output Factors though (and even then, only if not all the outputs are in a named chain).
            """
            # reconcile_direction('successor', previous_graph[c], current_graph[current_c], new_level, component_map)
        if len(new_level) > 0:
            return FactorGraph._reconcile_graph(new_level, component_map, current_graph, previous_graph)

    def load_graph(self, graph_file):
        """
        Method to load back in a graph. The graph file should be saved down using the save method, and is a JSON representation of the graph
        generated by the [networkx](https://networkx.github.io) library.

        :param graph_file: The file containing the primary model to load back for this inference algorithm.
        :type graph_file: str of filename
        """
        import json
        from ..util.graph_serialization import ModelComponentDecoder
        with open(graph_file) as f:
            json_graph = json.load(f, cls=ModelComponentDecoder)

        components_graph = nx.readwrite.json_graph.node_link_graph(
            json_graph, directed=True)
        components = {node.uuid: node for node in components_graph.nodes()}
        for node in components_graph.nodes():
            node._parent_graph = components_graph
            node.attributes = [components[a] for a in node.attributes]
        self._components_graph = components_graph
        for node in self._components_graph.nodes():
            if node.name is not None:
                self.__setattr__(node.name, node)
        return self

    def save(self, graph_file):
        """
        Method to save this graph down into a file. The graph file will be saved down as a JSON representation of the graph generated by the
        [networkx](https://networkx.github.io) library.

        :param graph_file: The file containing the primary model to load back for this inference algorithm.
        :type graph_file: str of filename
        """
        json_graph = nx.readwrite.json_graph.node_link_data(self._components_graph)
        import json
        from ..util.graph_serialization import ModelComponentEncoder
        with open(graph_file, 'w') as f:
            json.dump(json_graph, f, ensure_ascii=False, cls=ModelComponentEncoder)
