import math

from NeuralNetwork.Node.BasicNodeClasses import NodeName, InputLinkedNode, OutputLinkedNode


class HiddenNode(InputLinkedNode, OutputLinkedNode):
    def __init__(self, bias: float = None, node_layer: int = None, node_number: int = None):
        super(HiddenNode, self).__init__(bias=bias)
        name = NodeName(HiddenNode, node_number, layer=node_layer)
        self.name = name
        self.output_links = []
