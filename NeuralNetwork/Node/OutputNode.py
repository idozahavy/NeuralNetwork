from NeuralNetwork.Node.BasicNodeClasses import CalculatingNode, NodeName


class OutputNode(CalculatingNode):

    def __init__(self, input_links: list = None, bias: float = None, node_number: int = None):
        super(OutputNode, self).__init__(input_links=input_links, bias=bias)
        self.name = NodeName(OutputNode, node_number=node_number)
