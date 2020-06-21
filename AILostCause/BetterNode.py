import copy
import time
from random import random

import sympy
from sympy import Symbol, threaded

from AILostCause.AIMath import SigmoidFrom0To1

MUTATE_COEFFICIENT = 0.4


class InputNodeLink:
    """
    The connection between two nodes.
    The factor number is the number to multiply the input node activation with.
    It is the strings attaching the nodes together
    """

    def __init__(self, input_node=None, factor: float = None):
        self.input_node = input_node
        if factor:
            self.factor = factor
        else:
            self.factor = (random() - 0.5) * 5


class Node:
    input_links: list
    bias: float
    need_recalc: bool

    def __init__(self, input_nodes: list = None, bias: float = None):
        self.input_links = []
        if input_nodes:
            for node in input_nodes:
                self.AddInputNode(node)
        if bias:
            self.bias = bias
        else:
            self.bias = random() - 0.5
        self.activation = None
        self.need_recalc = True

    def __iter__(self, is_first=True):
        for link in self.input_links:
            yield link.input_node

    def __str__(self):
        string = f"Node - Bias={'{:5f}'.format(self.bias)},Links=["
        count = 1
        for link in self.input_links:
            string += f"Link{'{}'.format(count)} - Factor={'{:2.2f}'.format(float(link.factor))},Input=({str(link.input_node)}),"
            count += 1
        if len(self.input_links) == 0:
            string += f"],Activation={'{:5f}'.format(self.activation)}"
        else:
            string = string[:-1] + "]"

        return string

    def IsInput(self):
        return len(self.input_links) == 0

    def AddInputNode(self, node, factor: float = None):
        node: Node
        link = InputNodeLink(node, factor=factor)
        self.input_links.append(link)
        self.ResetActivation()

    def ResetActivation(self):
        self.activation = None
        self.need_recalc = True

    def SetActivation(self, activation: float):
        self.activation = activation
        self.need_recalc = True

    def SetToRecalculate(self):
        for link in self.input_links:
            link: InputNodeLink
            link.input_node.SetToRecalculate()
        self.need_recalc = True

    def NeedRecalculate(self, recursive=True) -> bool:
        if self.need_recalc:
            return True

        for link in self.input_links:
            link: InputNodeLink
            if recursive:
                self.need_recalc = link.input_node.NeedRecalculate()
            else:
                self.need_recalc = link.input_node.need_recalc
            if self.need_recalc:
                break
        return self.need_recalc

    def CalcActivation(self, recalculate=False, head=True) -> float:
        if recalculate:
            self.SetToRecalculate()
        elif head:
            self.NeedRecalculate(recursive=True)

        if self.need_recalc and len(self.input_links) > 0:
            activation = self.bias
            for link in self.input_links:
                link: InputNodeLink
                activation += link.factor * link.input_node.CalcActivation(head=False)
            self.activation = SigmoidFrom0To1(activation)

        self.need_recalc = False

        return self.activation

    def GetLinkToNode(self, node):
        node: Node
        for link in self.input_links:
            link: InputNodeLink
            if link.input_node == node:
                return link

    def FormulateFactorNodeActivation(self, formulate_link) -> Symbol:
        formula = self.bias
        if len(self.input_links) == 0:
            return self.activation
        for link in self.input_links:
            link: InputNodeLink
            if link == formulate_link:
                formula += Symbol("x") * link.input_node.FormulateFactorNodeActivation(None)
            else:
                formula += link.factor * link.input_node.FormulateFactorNodeActivation(formulate_link)
        formula = SigmoidFrom0To1(formula)
        return formula

    def FormulateBiasNodeActivation(self, formulate_node) -> Symbol:
        if self == formulate_node:
            activation = Symbol("x")
        else:
            activation = self.bias
        if len(self.input_links) == 0:
            return self.activation
        for link in self.input_links:
            link: InputNodeLink
            activation += link.factor * link.input_node.FormulateBiasNodeActivation(formulate_node)
        activation = SigmoidFrom0To1(activation)
        return activation

    def GetBackpropagationDeviationValue(self, formula, backprop_to, correct_output) -> float:
        formula = (formula - correct_output) ** 2
        output_deviation = correct_output - self.CalcActivation()
        if not formula.is_Float:

            x = Symbol("x")
            derivative = formula.diff(x)  # 0.01!!!
            # TODO same this derivative for latter calculations , because it takes a long time to make resets the same when Mutating

            #
            if isinstance(backprop_to, InputNodeLink):
                slope_on_point = derivative.subs(x, backprop_to.factor)
            elif isinstance(backprop_to, Node):
                slope_on_point = derivative.subs(x, self.bias)
            else:
                raise Exception("backprop was not InputNodeLink nor Node")
            wanted_change_from_slope = output_deviation / slope_on_point
            deviation_value = SigmoidFrom0To1(wanted_change_from_slope) - 0.5
            # 0.001

            return deviation_value
        return 0

    def GetBackpropagationFactorDeviationValue(self, backprop_link: InputNodeLink, correct_output):
        formula = self.FormulateFactorNodeActivation(backprop_link)  # 0.001
        deviation_value = self.GetBackpropagationDeviationValue(formula, backprop_link, correct_output)  # 0.01
        return deviation_value

    def GetBackpropagationBiasDeviationValue(self, backprop_node: InputNodeLink, correct_output):

        formula = self.FormulateBiasNodeActivation(backprop_node)  # 0.001

        deviation_value = self.GetBackpropagationDeviationValue(formula, backprop_node, correct_output)  # 0.01

        return deviation_value

    def GetBackpropagationDeviationNode(self, correct_output, output_node=None):

        if len(self.input_links) == 0:
            return self

        if not output_node:
            output_node = self

        deviation_links = []
        for link in self.input_links:
            input_deviation_node = link.input_node.GetBackpropagationDeviationNode(correct_output,
                                                                                   output_node=output_node)
            input_factor_deviation = output_node.GetBackpropagationFactorDeviationValue(link,
                                                                                        correct_output)  # 0.01 seconds
            deviation_link = InputNodeLinkDeviation(input_node=input_deviation_node, factor=link.factor,
                                                    factor_deviation=input_factor_deviation)
            deviation_links.append(deviation_link)

        node_bias_deviation = output_node.GetBackpropagationBiasDeviationValue(self,
                                                                               correct_output)  # 0.0085 seconds

        deviation_node = NodeDeviation(bias=self.bias, bias_deviation=node_bias_deviation)

        if deviation_links:
            deviation_node.input_links = deviation_links
        else:
            deviation_node.activation = self.activation

        return deviation_node


class InputNodeLinkDeviation(InputNodeLink):
    factor_deviation: float

    def __init__(self, input_node: Node = None, factor: float = None, factor_deviation: float = 0):
        super().__init__(input_node=input_node, factor=factor)
        self.factor_deviation = factor_deviation

    def AddDeviations(self, other):
        if not isinstance(other, InputNodeLink):
            return None
        if not isinstance(self.input_node, Node):
            return None
        if isinstance(other, InputNodeLinkDeviation):
            self.factor_deviation += other.factor_deviation
        self.input_node: NodeDeviation
        if not self.input_node.IsInput():
            self.input_node.AddDeviations(other.input_node)

    def DeleteDeviations(self):
        self.factor_deviation = 0

    def Mutate(self):
        self.factor += self.factor_deviation * MUTATE_COEFFICIENT
        if isinstance(self.input_node, NodeDeviation):
            self.input_node.Mutate()

    def MutateRandom(self, variation: float):
        self.factor += (2 * random() - 1) * variation
        if isinstance(self.input_node, NodeDeviation):
            self.input_node.MutateRandom()


class NodeDeviation(Node):
    bias_deviation: float

    def __init__(self, input_nodes: list = None, bias: float = None, bias_deviation: float = 0):
        super().__init__(input_nodes=input_nodes, bias=bias)
        self.bias_deviation = bias_deviation

    def __str__(self):
        string = f"Node - Bias={'{:2.2f}'.format(self.bias)}+({'{:2.2f}'.format(float(self.bias_deviation))}),Links=["
        count = 1
        for link in self.input_links:
            string += f"Link{'{}'.format(count)} - Factor={'{:2.2f}'.format(link.factor)}+({'{:2.2f}'.format(float(link.factor_deviation))}),Input=({str(link.input_node)}),"
            count += 1
        if len(self.input_links) == 0:
            string += f"],Activation={'{:2.2f}'.format(self.activation)}"
        else:
            string = string[:-1] + "]"

        return string

    def AddDeviations(self, other):
        if not isinstance(other, NodeDeviation):
            if isinstance(other, Node):
                new_other = NodeDeviation()
                new_other.input_links = other.input_links
                new_other.bias = other.bias
                new_other.activation = other.activation
                new_other.need_recalc = other.need_recalc
                other = new_other
            else:
                return None

        # this node bias addition
        self.bias_deviation += other.bias_deviation

        # links additions
        for link_index in range(len(self.input_links)):
            if not isinstance(self.input_links[link_index], InputNodeLinkDeviation):
                return None
            self.input_links[link_index].AddDeviations(other.input_links[link_index])

        return self

    def DeleteDeviations(self):
        self.bias_deviation = 0
        for link in self.input_links:
            link.DeleteDeviations()

    def Mutate(self):
        self.bias += self.bias_deviation * MUTATE_COEFFICIENT
        self.need_recalc = True
        for link in self.input_links:
            link: InputNodeLinkDeviation
            link.Mutate()
        self.DeleteDeviations()

    def MutateRandom(self, variation: float):
        self.bias += (2 * random() - 1) * variation
        self.need_recalc = True
        for link in self.input_links:
            link: InputNodeLinkDeviation
            link.MutateRandom(variation)
