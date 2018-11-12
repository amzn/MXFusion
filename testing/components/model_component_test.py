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


import unittest
import networkx as nx
import mxfusion.components as mfc


class ModelComponentTests(unittest.TestCase):
    """
    Tests the MXFusion.core.model_component.ModelComponent class.
    """

    def test_switch_simple_backwards(self):

        node_a = mfc.ModelComponent()
        node_b = mfc.ModelComponent()
        node_c = mfc.ModelComponent()
        node_a.predecessors = [('edge_1', node_b), ('edge_2', node_c)]
        node_b.successors = [('edge_1', node_a)]
        node_c.successors = [('edge_2', node_a)]
        #
        # successors = set([(node_b.uuid, node_a.uuid), (node_c.uuid, node_a.uuid)])

        graph = nx.MultiDiGraph()

        node_a.graph = graph

        self.assertTrue(set([v for _, v in node_b.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_c.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_a.predecessors]) == set([node_c, node_b]))

    def test_switch_simple_forwards(self):

        node_a = mfc.ModelComponent()
        node_b = mfc.ModelComponent()
        node_c = mfc.ModelComponent()
        node_a.predecessors = [('edge_1', node_b), ('edge_2', node_c)]
        node_b.successors = [('edge_1', node_a)]
        node_c.successors = [('edge_2', node_a)]

        graph = nx.MultiDiGraph()

        node_c.graph = graph

        self.assertTrue(set([v for _, v in node_b.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_c.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_a.predecessors]) == set([node_c, node_b]))

    def test_switch_multilayer(self):

        node_a = mfc.ModelComponent()
        node_b = mfc.ModelComponent()
        node_c = mfc.ModelComponent()
        node_d = mfc.ModelComponent()
        node_e = mfc.ModelComponent()
        node_a.predecessors = [('edge_1', node_b), ('edge_2', node_c)]
        node_b.predecessors = [('edge_1', node_d), ('edge_2', node_e)]

        graph = nx.MultiDiGraph()

        node_a.graph = graph

        self.assertTrue(set([v for _, v in node_b.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_c.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_a.predecessors]) == set([node_c, node_b]))

        self.assertTrue(set([v for _, v in node_d.successors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_e.successors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_b.predecessors]) == set([node_d, node_e]))

    def test_join_attach_new_successor_not_to_graph(self):

        node_a = mfc.ModelComponent()
        graph = nx.MultiDiGraph()

        node_b = mfc.ModelComponent()
        node_d = mfc.ModelComponent()
        node_e = mfc.ModelComponent()
        node_b.predecessors = [('edge_1', node_d), ('edge_2', node_e)]
        node_b.graph = graph

        node_c = mfc.ModelComponent()
        node_a.predecessors = [('edge_1', node_b), ('edge_2', node_c)]

        self.assertTrue(set([v for _, v in node_b.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_c.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_a.predecessors]) == set([node_c, node_b]))

        self.assertTrue(set([v for _, v in node_d.successors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_e.successors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_b.predecessors]) == set([node_d, node_e]))


    def test_join_predecessors_not_in_graph_to_node_in_graph(self):

        node_a = mfc.ModelComponent()
        graph = nx.MultiDiGraph()
        node_a.graph = graph

        node_b = mfc.ModelComponent()
        node_d = mfc.ModelComponent()
        node_e = mfc.ModelComponent()
        node_b.predecessors = [('edge_1', node_d), ('edge_2', node_e)]

        node_a.predecessors = [('edge_1', node_b)]

        self.assertTrue(set([v for _, v in node_b.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_a.predecessors]) == set([node_b]))

        self.assertTrue(set([v for _, v in node_d.successors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_e.successors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_b.predecessors]) == set([node_d, node_e]))

    def test_join_successors_not_in_graph_to_node_in_graph(self):
        node_a = mfc.ModelComponent()
        node_b = mfc.ModelComponent()
        node_d = mfc.ModelComponent()
        node_e = mfc.ModelComponent()
        node_b.successors = [('edge_1', node_d), ('edge_2', node_e)]
        graph = nx.MultiDiGraph()

        node_a.graph = graph
        node_a.successors = [('edge_1', node_b)]

        self.assertTrue(set([v for _, v in node_a.successors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_b.predecessors]) == set([node_a]))

        self.assertTrue(set([v for _, v in node_d.predecessors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_e.predecessors]) == set([node_b]))
        self.assertTrue(set([v for _, v in node_b.successors]) == set([node_d, node_e]))

    def test_multiple_successors_same_name(self):
        node_a = mfc.ModelComponent()
        node_b = mfc.ModelComponent()
        node_c = mfc.ModelComponent()
        node_a.predecessors = [('edge_1', node_b), ('edge_1', node_c)]

        graph = nx.MultiDiGraph()

        node_a.graph = graph

        self.assertTrue(set([v for _, v in node_b.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_c.successors]) == set([node_a]))
        self.assertTrue(set([v for _, v in node_a.predecessors]) == set([node_c, node_b]))
