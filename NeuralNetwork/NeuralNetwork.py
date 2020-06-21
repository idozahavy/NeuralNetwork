import copy
import time

from NeuralNetwork.Node.BasicNodeClasses import NodeLink, InputLinkedNode, OutputLinkedNode, NodeName
from NeuralNetwork.Node.HiddenNode import HiddenNode
from NeuralNetwork.Node.InputNode import InputNode
from NeuralNetwork.Node.OutputNode import OutputNode


class NeuralNetwork:
    input_nodes: list
    output_nodes: list
    hidden_nodes: list
    node_dictionary: dict

    def __init__(self, input_count, output_count, hidden_layer_count, hidden_layer_nodes_count):  # 0.127 secs once for (784, 10, 2, 16)

        self.input_nodes = []
        self.output_nodes = []
        self.hidden_nodes = []
        self.node_dictionary = {}

        for index in range(input_count):
            new_input_node = InputNode(activation=None, node_number=index)
            self.input_nodes.append(new_input_node)
            self.node_dictionary[str(new_input_node)] = new_input_node

        next_input_nodes = self.input_nodes

        for layer_index in range(hidden_layer_count):
            current_input_nodes = next_input_nodes
            next_input_nodes = []
            for node_index in range(hidden_layer_nodes_count):
                new_hidden_node = HiddenNode(bias=None, node_number=node_index, node_layer=layer_index)
                for input_node in current_input_nodes:
                    input_node: OutputLinkedNode
                    new_link = NodeLink(input_node, new_hidden_node, factor=None)
                    input_node.AddOutputLink(new_link)
                    new_hidden_node.AddInputLink(new_link)
                next_input_nodes.append(new_hidden_node)
                self.hidden_nodes.append(new_hidden_node)
                self.node_dictionary[str(new_hidden_node)] = new_hidden_node

        for index in range(output_count):
            new_output_node = OutputNode(bias=None, node_number=index)

            for input_node in next_input_nodes:
                input_node: OutputLinkedNode
                new_link = NodeLink(input_node, new_output_node, factor=None)
                input_node.AddOutputLink(new_link)
                new_output_node.AddInputLink(new_link)

            self.output_nodes.append(new_output_node)
            self.node_dictionary[str(new_output_node)] = new_output_node

    def __str__(self):
        string = "NeuralNetwork\n"
        for input_node in self.input_nodes:
            string += str(input_node)
        last_layer = -1
        for hidden_node in self.hidden_nodes:
            hidden_node: HiddenNode
            if hidden_node.name.layer != last_layer:
                string += "\n"
            string += str(hidden_node)
            last_layer = hidden_node.name.layer
        string += "\n"
        for output_node in self.output_nodes:
            string += str(output_node)
        return string

    def GetOutputs(self, input_activation_list: list):  # 0.018 secs once for (784, 10, 2, 16)
        input_activation_list = copy.deepcopy(input_activation_list)

        self.ResetActivations()

        for input_node in self.input_nodes:
            input_node: InputNode
            input_node.SetActivation(input_activation_list.pop())

        output_activations = []

        for output_node in self.output_nodes:
            output_node: OutputNode
            output_activation = output_node.GetActivation(save=True, recalculate=False)
            output_activations.append(output_activation)

        return output_activations

    def GetHiddenNode(self, layer: int, node_number: int):
        name = NodeName(HiddenNode, node_number, layer=layer)
        return self.node_dictionary[str(name)]

    def ResetActivations(self):
        for output_node in self.output_nodes:
            output_node: OutputNode
            output_node.ResetActivation()
        for hidden_node in self.hidden_nodes:
            hidden_node: HiddenNode
            hidden_node.ResetActivation()
