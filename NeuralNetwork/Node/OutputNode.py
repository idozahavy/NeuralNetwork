from NeuralNetwork.Node.BasicNodeClasses import NodeName, InputLinkedNode


class OutputNode(InputLinkedNode):

    def __init__(self, bias: float = None, node_number: int = None):
        super(OutputNode, self).__init__(bias=bias)
        self.name = NodeName(OutputNode, node_number, layer=None)
