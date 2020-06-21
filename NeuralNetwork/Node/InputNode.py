import math

from NeuralNetwork.Node.BasicNodeClasses import NodeName, OutputLinkedNode


class InputNode(OutputLinkedNode):
    def __init__(self, activation: float = None, node_number: int = None):
        if activation is None:
            activation = math.nan
        super(InputNode, self).__init__(activation=activation)
        name = NodeName(InputNode, node_number, layer=None)
        self.name = name
