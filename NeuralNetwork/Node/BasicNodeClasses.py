import math
import time
from abc import ABCMeta, abstractmethod, ABC
from random import random

from NeuralNetwork.Node.NodeMath import Sigmoid


class NodeName:
    node_type: str
    layer: int
    node_number: int

    def __init__(self, node_type: type, node_number: int, layer: int = None):
        self.node_type = node_type.__name__.split('.')[-1]
        self.layer = layer
        self.node_number = node_number

    def __eq__(self, other):
        if isinstance(other, NodeName):
            other: NodeName
            if self.node_type == other.node_type:
                if self.layer == other.layer:
                    if self.node_number == other.node_number:
                        return True
        return False

    def __str__(self):
        string = "(" + self.node_type
        if "Hidden" in self.node_type:
            string += f" L{self.layer}"
        if self.node_number is None:
            string += " Undefined Number"
        else:
            string += f" N{self.node_number}"
        return string + ")"


class Node(ABC):
    activation: float
    name: NodeName

    @abstractmethod
    def __init__(self, activation=None):
        self.activation = activation
        self.name = None

    @abstractmethod
    def GetActivation(self) -> float:
        return self.activation

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name


class InputLinkedNode(Node):
    input_links: list
    bias: float

    @abstractmethod
    def __init__(self, bias: float = None):
        super(InputLinkedNode, self).__init__(activation=math.nan)
        self.input_links = []
        if bias:
            self.bias = bias
        else:
            # self.bias = (2*random()-1)
            self.bias = 0
        self.cost_value_der = None
        self.cost_bias_der = None

    def GetActivation(self, save: bool = False, recalculate: bool = False):
        if not recalculate and math.isfinite(self.activation):
            return self.activation

        activation = self.bias
        for input_link in self.input_links:
            input_link: NodeLink

            if isinstance(input_link.input_node, InputLinkedNode):  # 0.017
                activation += input_link.factor * input_link.input_node.GetActivation(save=save, recalculate=recalculate)
            else:
                activation += input_link.factor * input_link.input_node.GetActivation()
            # activation += input_link.GetLinkActivation(save=save, recalculate=recalculate)  # 0.021

        activation = Sigmoid(activation)
        if save:
            self.activation = activation
        return activation

    def ResetActivation(self):
        self.activation = math.nan

    def AddInputLink(self, link):
        if not self.input_links:
            self.input_links = []
        if link not in self.input_links:
            self.input_links.append(link)
        else:
            raise Exception(f"Tried adding input link '{link}' to InputLinkedNode '{str(self.name)}' that already exist")


class OutputLinkedNode(Node):
    output_links: list

    @abstractmethod
    def __init__(self, activation: float = None):
        super(OutputLinkedNode, self).__init__(activation=activation)
        self.output_links = []

    def GetActivation(self):
        if "Input" not in str(self.name.node_type):
            print("fuck me - error 101")

        if isinstance(self.activation, float) and not math.isfinite(self.activation):
            raise Exception(f"OutputLinkedNode named '{self.name}' do not have any activation set")
        return self.activation

    def SetActivation(self, activation):
        self.activation = activation

    def AddOutputLink(self, link):
        if not self.output_links:
            self.output_links = []
        if link not in self.output_links:
            self.output_links.append(link)
        else:
            raise Exception(f"Tried adding output link '{link}' to OutputLinkedNode '{str(self.name)}' that already exist")


class NodeLink:
    input_node: OutputLinkedNode
    output_node: InputLinkedNode
    factor: float

    def __init__(self, input_node, output_node, factor: float = None):
        self.input_node = input_node
        self.output_node = output_node
        if factor is not None:
            self.factor = factor
        else:
            self.factor = (2*random()-1)
        self.cost_factor_der = None

    def GetLinkActivation(self, save: bool = False, recalculate: bool = False):
        if isinstance(self.input_node, InputLinkedNode):
            return self.factor * self.input_node.GetActivation(save=save, recalculate=recalculate)
        else:
            return self.factor * self.input_node.GetActivation()
