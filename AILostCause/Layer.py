import copy

from AILostCause.Node import Node


class Layer:
    input_count = 1
    node_list = []

    def __init__(self, input_count, node_count=0):
        self.input_count = input_count
        self.node_list = []
        if node_count:
            self.Randomize(node_count)

    def __len__(self):
        return len(self.node_list)

    def GetOutput(self, input_list: list):
        if len(input_list) != self.input_count:
            return None
        current_output = []
        for node in self.node_list:
            node_result = node.GetOutputs(input_list)
            current_output.append(node_result)
        return current_output

    def Randomize(self, node_count):
        self.node_list = [Node(self.input_count, randomize=True) for _ in range(node_count)]

    def CreateChild(self, variation: float):
        child = copy.deepcopy(self)
        child.Mutate(variation)
        return child

    def Mutate(self, variation: float):
        for node in self.node_list:
            node.Mutate(variation)

    def DeviateWithMatrix(self, matrix):
        for index in range(len(self.node_list)):
            self.node_list[index].DeviateWithFactorList(matrix[index])

    def BackpropagationDeviationMatrix(self, input_list: list, output_deviation_list):

        layer_deviation = []

        for node_index in range(len(self.node_list)):
            output_deviation = output_deviation_list[node_index]
            node_back = self.node_list[node_index].BackpropagationDeviationList(input_list, output_deviation)
            layer_deviation.append(node_back)
        return layer_deviation

    def __str__(self):
        string = "["
        node_count = 0
        for node in self.node_list:
            string += f"Node_{str(node_count)} {str(node)}, "
            node_count += 1
        string = string[:-2]
        string += "]"
        return string

