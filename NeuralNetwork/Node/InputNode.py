import math

from NeuralNetwork.Node.BasicNodeClasses import Node, NodeName


class InputNode(Node):
    output_links: list

    def __init__(self, activation: float = None, node_number: int = None, output_links: list = None):
        name = NodeName(InputNode, node_number=node_number)
        super(InputNode, self).__init__(activation=activation, name=name)
        self.output_links = output_links

    def GetActivation(self):
        if not math.isfinite(self.activation):
            raise Exception(f"Input node name '{self.name}' do not have any activation set")
        return self.activation

    def SetActivation(self, activation):
        self.activation = activation

    def AddConfiguredOutputLink(self, link):
        if link not in self.output_links:
            self.output_links.append(link)
        else:
            raise Exception(f"Tried adding output link '{link}' to input '{str(self.name)}' that already exist")
