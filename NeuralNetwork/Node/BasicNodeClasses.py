import math
from abc import ABCMeta, abstractmethod, ABC
from random import random

from NeuralNetwork.Node.NodeMath import SigmoidFrom0To1


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
    def __init__(self, activation=None, name: NodeName = None):
        self.activation = activation
        self.name = name

    @abstractmethod
    def GetActivation(self) -> float:
        return self.activation

    def __str__(self):
        return str(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name


class CalculatingNode(Node):
    input_links: list
    bias: float

    @abstractmethod
    def __init__(self, input_links: list = None, bias: float = None):
        super(CalculatingNode, self).__init__()
        self.input_links = input_links
        if not self.input_links:
            self.input_links = []
        if bias:
            self.bias = bias
        else:
            self.bias = (2*random()-1)
        self.activation = math.nan

    def GetActivation(self, save: bool = False, recalculate: bool = False):
        if not recalculate and math.isfinite(self.activation):
            return self.activation

        activation = self.bias
        for input_link in self.input_links:
            input_link: NodeLink
            activation += input_link.GetLinkActivation()
        activation = SigmoidFrom0To1(activation)
        if save:
            self.activation = activation
        return activation

    def ResetActivation(self):
        self.activation = math.nan


class NodeLink:
    input_node: Node
    output_node: Node
    factor = None

    def __init__(self, input_node, output_node, factor=None):
        self.input_node = input_node
        self.output_node = output_node
        if factor:
            self.factor = factor
        else:
            self.factor = (2*random()-1)

    def GetLinkActivation(self, save: bool = False, recalculate: bool = False):
        if isinstance(self.input_node, CalculatingNode):
            return self.factor * self.input_node.GetActivation(save=save, recalculate=recalculate)
        return self.factor * self.input_node.GetActivation()

