import math

from NeuralNetwork.Node.BasicNodeClasses import CalculatingNode, NodeName


class HiddenNode(CalculatingNode):
    output_links: list

    def __init__(self, input_links: list = None, output_links: list = None, bias: float = None,
                 node_layer: int = None, node_number: int = None):
        super().__init__(input_links=input_links, bias=bias)
        name = NodeName(HiddenNode, node_number, layer=node_layer)
        self.name = name
        self.output_links = output_links
        if not self.output_links:
            self.output_links = []
        self.activation = math.nan

