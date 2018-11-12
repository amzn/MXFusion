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


from uuid import uuid4
from ..common.exceptions import ModelSpecificationError


class ModelComponent(object):
    """
    The building block of a Model in MXFusion.

    ModelComponents exist in one of two modes.

    **Mode 1 - Bi-directional mode**

    If a node is not attached to a FactorGraph, it maintains a list of all of its predecessors and successors directly.
    These are stored in the ``self._predecessors`` and ``self._successors`` methods.

    **Mode 2 - Graph mode**

    If a node is attached to a FactorGraph, it does not store direct references to its successors and predecessors.
    When accessed, the predecessors/successors properties directly query the graph they are attached to to find out what the respective neighbor nodes are.
    """

    def __init__(self):
        self.name = None
        self._uuid = str(uuid4()).replace('-', '_')
        self._parent_graph = None
        self._successors = []  # either [('name', Variable), ('factor', Factor), ...]
        self._predecessors = []
        self.attributes = []

    @property
    def uuid(self):
        """
        Return the UUID of this graph
        """
        return self._uuid

    def __hash__(self):
        return self._uuid.__hash__()

    def __eq__(self, other):
        return self._uuid.__hash__() == other.__hash__()

    def __repr__(self):
        return self.uuid

    def as_json(self):
        return {'uuid': self._uuid,
                'name': self.name,
                'attributes': [a.uuid for a in self.attributes]}

    @property
    def graph(self):
        """
        Return the object's graph
        """
        return self._parent_graph

    @graph.setter
    def graph(self, graph):
        """
        Attaches the node to a graph, switching from Bidirectional mode to Graph mode if it is not already in Graph mode.

        A node cannot be re-attached to a different graph once it is attached. Use the ``replicate()`` functionality if you need to do this.

        :param graph: The ``components_graph`` of the ``FactorGraph`` this node is attaching to.
        :type graph: networkx.DiGraph
        """
        if self._parent_graph is not None:
            if self._parent_graph == graph:
                return
            elif graph is not None:
                raise ModelSpecificationError("Trying to reset a variables graph is bad!")
        self._parent_graph = graph
        if self._parent_graph is not None:
            self._parent_graph.add_node(self)

        self.predecessors = self._predecessors
        self.successors = self._successors
        self._update_attributes()

        self._predecessors = []
        self._successors = []

    def _update_attributes(self):
        """
        Adds all of a node's attributes to its graph.
        """
        for a in self.attributes:
            self.graph.add_node(a)

    def _align_graph_modes(self, edge_nodes):
        """
        This function will update the current node and all nodes passed in to be in Graph mode if any of edge_nodes are in Graph mode.

        :param edge_nodes: All the nodes to align to the same graph mode. I.E. predecessors or successors.
        :type edge_nodes: List of tuples of name to node e.g. [('random_variable': Variable y)]
        """

        if self.graph is None and any([node.graph is not None for _, node in edge_nodes]):
            # Put self into the graph, put all the other nodes in that graph (if more than one just error).
            graphs = set([node.graph for _, node in edge_nodes if node.graph is not None])
            if len(graphs) > 1:
                raise ModelSpecificationError("Too many graphs!")
            graph = list(graphs)[0]
            self.graph = graph
            for _, node in edge_nodes:
                node.graph = graph

    @property
    def successors(self):
        """
        Return a list of nodes pointed to by the edges of this node.

        Note: The ordering of this list is not guaranteed to be consistent with assigned order.
        """
        if self.graph is not None:
            succ = [(e['name'], v) for v, edges in self.graph.succ[self].items() for e in edges.values()]
            return succ
        else:
            return self._successors

    @successors.setter
    def successors(self, successors):
        """
        Sets this node's successors to those passed in.

        :param successors: List of tuples of name to node e.g. [('random_variable': Variable y)].
        :type successors: List of tuples of name to node e.g. [('random_variable': Variable y)]
        """
        def add_predecessor(successor, predecessor, successor_name):
            if successor.graph is None:
                successor._predecessors.append((successor_name, predecessor))
            if successor.graph is not None:
                raise ModelSpecificationError("Internal Error. Cannot add predecessor when a component is attached to a graph.")

        self._align_graph_modes(successors)
        if self.graph is not None:
            for _, successor in self.successors:
                self.graph.remove_edge(self, successor)
            for name, successor in successors:
                successor.graph = self.graph
                self.graph.add_edge(self, successor, key=name, name=name)
        else:
            self._successors = successors
            for name, successor in successors:
                add_predecessor(successor, self, name)


    @property
    def predecessors(self):
        """
        Return a list of nodes whose edges point into this node.

        Note: The ordering of this list is not guaranteed to be consistent with assigned order.
        """
        if self.graph is not None:
            pred = [(e['name'], v) for v, edges in self.graph.pred[self].items() for e in edges.values()]
            return pred
        else:
            return self._predecessors

    @predecessors.setter
    def predecessors(self, predecessors):
        """
        Sets this node's predecessors to be those passed in.

        :param predecessors: List of tuples of name to node e.g. [('random_variable': Variable y)]
        :type predecessors: List of tuples of name to node e.g. [('random_variable': Variable y)]
        """
        def add_successor(predecessor, successor, predecessor_name):
            if predecessor.graph is None:
                predecessor._successors.append((predecessor_name, successor))
            if predecessor.graph is not None:
                raise ModelSpecificationError("Internal Error. Cannot add a successor when a component is attached to a graph.")

        self._align_graph_modes(predecessors)
        if self.graph is not None:
            for _, predecessor in self.predecessors:
                self.graph.remove_edge(predecessor, self)
            for name, predecessor in predecessors:
                predecessor.graph = self.graph
                self.graph.add_edge(predecessor, self, key=name, name=name)
        else:
            self._predecessors = predecessors
            for name, predecessor in predecessors:
                add_successor(predecessor, self, name)

    def _replicate_self_with_attributes(self, var_map):
        """
        Replicates self if not in ``var_map``. Also replicates all of self's attributes.

        :param var_map: A mapping from the original model's components to the replicated components.
        :type var_map: {original_node: new_node}
        :rtype: ModelComponent
        """
        var_map = var_map if var_map is not None else {}
        if self in var_map:
            return var_map[self]

        attributes_map = {}
        for a in self.attributes:
            if a in var_map:
                attributes_map[a] = var_map[a]
            else:
                attributes_map[a] = a.replicate_self()
                var_map[a] = attributes_map[a]
        replicated_component = self.replicate_self(attributes_map)
        var_map[self] = replicated_component
        return replicated_component

    def _replicate_neighbors(self, var_map, neighbors, recurse_type, replication_function):
        """
        Helper function that returns a replicated list of neighbors based on the recurse_type passed in.

        :param var_map: A mapping from the original model's components to the replicated components.
        :type var_map: {original_node: new_node}
        :param neighbors: Dictionary containing the list of a node's neighbors in one direction (predecessors or successors).
        :type neighbors: List of tuples of name to node e.g. [('random_variable': Variable y)]
        :param recurse_type: Parameter that decides how to replicate the neighbor nodes. Must be one of: 'recursive', 'one_level', or None.
        :type recurse_type: String or None
        :param replication_function: A function that takes in a ModelComponent and returns an answer for how to replicate that node's predecessors and successors.
        :type replication_function: function

        """
        if recurse_type == 'recursive':
            replicated_neighbors = [(name, i.replicate(var_map=var_map, replication_function=replication_function))
                                   for name, i in neighbors]
        elif recurse_type == 'one_level':
            replicated_neighbors = [(name, i._replicate_self_with_attributes(var_map=var_map))
                                   for name, i in neighbors]
        elif recurse_type is None:
            replicated_neighbors = []
        else:
            raise ModelSpecificationError("Parameter 'recurse_type' must be 'recursive', 'one_level', or None.")
        return replicated_neighbors

    def replicate(self, var_map=None, replication_function=None):
        """
        Replicates this component and its neighbors based on the replication_function logic passed in.

        :param var_map: A mapping from the original model's components to the replicated components. This is used to track which components
            have already been replicated in a dynamic programming style.
        :type var_map: {original_node: new_node}
        :param replication_function: A function that takes in a ModelComponent and returns an answer for how to replicate that node's predecessors and successors. If None, only replicates this node.
        :type replication_function: function
        """
        var_map = var_map if var_map is not None else {}
        if self in var_map:
            return var_map[self]
        replicated_component = self._replicate_self_with_attributes(var_map)

        if replication_function is not None:
            pred_recursion, succ_recursion = replication_function(self)
        else:
            pred_recursion, succ_recursion = None, None
        predecessors = self._replicate_neighbors(var_map, self.predecessors, pred_recursion, replication_function)
        successors = self._replicate_neighbors(var_map, self.successors, succ_recursion, replication_function)
        replicated_component.predecessors = predecessors
        replicated_component.successors = successors

        return replicated_component
