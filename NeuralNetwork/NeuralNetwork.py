import copy
import pickle
import time
from random import random

from NeuralNetwork.Node.BasicNodeClasses import NodeLink, InputLinkedNode, OutputLinkedNode, NodeName
from NeuralNetwork.Node.HiddenNode import HiddenNode
from NeuralNetwork.Node.InputNode import InputNode
from NeuralNetwork.Node.NodeMath import SigmoidDerivative
from NeuralNetwork.Node.OutputNode import OutputNode


def LoadFromFile(file_path):
    with open(file_path, mode='rb') as file:
        obj = pickle.load(file)
        file.close()
        return obj


class NeuralNetwork:
    input_nodes: list
    output_nodes: list
    hidden_nodes: list
    node_dictionary: dict

    def __init__(self, input_count: int, output_count: int, hidden_layer_count: int,
                 hidden_layer_nodes_count: int):  # 0.127 secs once for (784, 10, 2, 16)

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

    def SetInputs(self, input_activation_list: list):
        input_activation_list = copy.deepcopy(input_activation_list)

        self.ResetActivations()

        for input_node in self.input_nodes:
            input_node: InputNode
            input_node.SetActivation(input_activation_list.pop(0))

    def GetOutputs(self, input_activation_list: list):  # 0.018 secs once for (784, 10, 2, 16)
        self.SetInputs(input_activation_list)

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

    def GetLoss(self, input_list: list, desired_output_list: list):
        self.SetInputs(input_list)
        total_loss = 0
        index = 0
        for output in self.output_nodes:
            output: InputLinkedNode
            total_loss += (output.GetActivation(save=True, recalculate=False) - desired_output_list[index]) ** 2
            index += 1
        return total_loss / len(self.output_nodes)

    def CalculateSlopeValuesFromCost(self, input_list: list, cost: float):
        """
        changes the most activate output by the cost
        if cost is positive it encourage this output
        if cost is negative it discourage this output
        :param input_list:
        :param cost:
        :return:
        """
        highest_activation_index = None
        highest_activation = 0

        for index in range(len(self.output_nodes)):
            output = self.output_nodes[index]
            output: OutputNode
            output_activation = output.GetActivation(save=True, recalculate=False)
            if output_activation > highest_activation:
                highest_activation = output_activation
                highest_activation_index = index

        desired_output_list = [output.GetActivation(save=True, recalculate=False) for output in self.output_nodes]
        desired_output_list[highest_activation_index] += cost

        self.CalculateSlopeValues(input_list, desired_output_list)

    def CalculateSlopeValues(self, input_list: list, desired_output_list: list):  # 0.034 once for (784, 10, 2, 16)
        self.GetOutputs(input_list)

        self._OutputSlopes(desired_output_list)
        self._HiddenSlopes()
        self._InputSlopes()

    def _OutputSlopes(self, desired_output_list: list):
        output_index = 0
        for output in self.output_nodes:
            output: OutputNode
            output_value_der = SigmoidDerivative(output.GetActivation(save=True, recalculate=False))  # d(Output)/d(Value)
            cost_output_der = (desired_output_list[output_index] - output.GetActivation(save=True, recalculate=False))  # d(Cost)/d(Output)

            cost_value_der = cost_output_der * output_value_der

            output.cost_value_der = cost_value_der
            output.cost_bias_der = cost_value_der  # d(Value)/d(Bias) = 1
            output_index += 1

    def _HiddenSlopes(self):
        for node in reversed(self.hidden_nodes):
            node: HiddenNode
            if node.cost_value_der is None:
                node.cost_value_der = 0
            if node.cost_bias_der is None:
                node.cost_bias_der = 0

            for link in node.output_links:
                link: NodeLink
                if link.cost_factor_der is None:
                    link.cost_factor_der = 0
                if link.output_node.cost_value_der is None:
                    raise Exception("Problem 404 - d_cost_d_sigmoid not found")

                value_factor_der = node.GetActivation(save=True, recalculate=False)  # d(Value)/d(Factor)
                cost_factor_der = link.output_node.cost_value_der * value_factor_der
                link.cost_factor_der += cost_factor_der

                value_output_1_der = link.factor
                cost_output_1_der = link.output_node.cost_value_der * value_output_1_der  # d(Cost)/d(Output(-1))

                output_1_value_1_der = SigmoidDerivative(node.GetActivation(save=True, recalculate=False))  # d(Output(-1))/d(Value(-1))
                cost_value_1_der = cost_output_1_der * output_1_value_1_der

                node.cost_value_der += cost_value_1_der

                node.cost_bias_der += cost_value_1_der  # d(Value)/d(Bias) = 1

    def _InputSlopes(self):
        for node in self.input_nodes:
            for link in node.output_links:
                link: NodeLink
                if link.cost_factor_der is None:
                    link.cost_factor_der = 0
                d_cost_d_factor = node.GetActivation()  # d(Sigmoid)/d(Factor)
                if link.output_node.cost_value_der is None:
                    raise Exception("Problem 404 - d_cost_d_sigmoid not found")
                d_cost_d_factor *= link.output_node.cost_value_der
                link.cost_factor_der += d_cost_d_factor

    def MutateSlopeValues(self, coefficient: float = 0.1):
        for node in self.node_dictionary.values():
            if isinstance(node, InputLinkedNode):
                node: InputLinkedNode
                for link in node.input_links:
                    link.factor += link.cost_factor_der * coefficient
                node.bias += node.cost_bias_der * coefficient
        self.ResetCosts()

    def MutateRandom(self, variation: float):
        for node in self.hidden_nodes and self.output_nodes:
            node: InputLinkedNode
            for link in node.input_links:
                link.factor += (2*random()-1) * variation
            node.bias += (2*random()-1) * variation

    def GetRandomizedBest(self, input_list: list, desired_output_list: list, variation: float, tries: int):
        best_loss = self.GetLoss(input_list, desired_output_list)
        best_child = None
        for _ in range(tries):
            child = copy.deepcopy(self)
            child.MutateRandom(variation)
            loss = child.GetLoss(input_list, desired_output_list)
            if loss < best_loss:
                best_child = child
                best_loss = loss

        if not best_child:
            return copy.deepcopy(self)
        return best_child

    def MakeChild(self, other_parent):
        other_parent: NeuralNetwork
        # TODO make a child that inherits values from each parent and random select between these values
        # Evolutionary Algorithm
        pass

    def ResetCosts(self):
        for node in self.node_dictionary.values():
            if isinstance(node, InputLinkedNode):
                node: InputLinkedNode
                node.cost_value_der = None
                node.cost_bias_der = None
                for link in node.input_links:
                    link.cost_factor_der = None

    def SaveToFile(self, file_path):
        with open(file_path, mode='wb') as file:
            pickle.dump(self, file, pickle.HIGHEST_PROTOCOL)
            file.close()

