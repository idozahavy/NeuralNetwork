import copy

from AILostCause.AIMath import MatricesMinus, LAYER_BACKPROPAGATION_COEFFICIENT
from AILostCause.Layer import Layer
from AILostCause.Node import Node


class AI:
    input_count = 1  # numbers
    output_count = 1  # numbers
    layer_list = []  # list of lists of Node
    evaluate_function = None

    def __init__(self, input_count: int, output_count: int,
                 evaluate_function=None, nodes_layers: tuple = (0, 0)):
        self.input_count = input_count
        self.output_count = output_count
        self.evaluate_function = evaluate_function
        self.score = 0
        if nodes_layers:
            self.RandomizeMatrix(nodes_layers[0], nodes_layers[1])

    def RandomizeMatrix(self, node_count, layers):
        self.layer_list = []
        current_input_count = self.input_count
        for i in range(layers):
            new_layer = Layer(current_input_count, node_count)
            self.layer_list.append(new_layer)
            current_input_count = len(new_layer)
        self.layer_list.append(Layer(current_input_count, self.output_count))

    def CreateChild(self, variation: float):
        child = copy.deepcopy(self)
        child.Mutate(variation)
        return child

    def Mutate(self, variation: float):
        for layer in self.layer_list:
            layer.Mutate(variation)

    def GetOutputMatrix(self, input_list: list):
        if len(input_list) != self.input_count:
            return None
        output_matrix = []
        layer_result = input_list
        for layer in self.layer_list:
            layer_result = layer.GetOutput(layer_result)
            output_matrix.append(layer_result)
        return output_matrix

    def GetOutput(self, input_list: list, layer_node=(-1, -1)):
        if len(input_list) != self.input_count:
            return None
        current_input = input_list
        current_output = []
        layer_current_index = 0
        for layer in self.layer_list:
            current_output = layer.GetOutput(current_input)
            current_input = current_output
            if layer_current_index == layer_node[0]:
                if layer_node[1] < 0:
                    return current_output
                else:
                    return current_output[layer_node[1]]
            layer_current_index += 1
        return current_output

    def GetLayerOutput(self, input_list: list, layer_index: int):
        return self.GetOutput(input_list, layer_node=(layer_index, -1))

    def GetNodeOutput(self, input_list: list, layer_index: int, node_index: int):
        return self.GetOutput(input_list, layer_node=(layer_index, node_index))

    def GetNodeWeight(self, input_list: list, layer_index: int, node_index: int):
        pass

    def GradientAdjuster(self):
        """
        takes an input and an output and checks the deviation from needed result.
        do the same for a minimized mutation(one parameter) and check if its a better deviation.
        if the deviation is big go fast if not go slow.
        continue until the parameter is at its best case.
        like a ball rolling on a hill.
        Backpropagation
        https://youtu.be/IHZwWFHWa-w?t=359
        :return:
        """
        pass

    def DeviateWithMatrixList(self, matrix_list):
        for index in range(len(self.layer_list)):
            self.layer_list[index].DeviateWithMatrix(matrix_list[index])

    def __str__(self):
        string = "AI\n"
        layer_count = 0
        for layer in self.layer_list:
            string += f"Layer_{str(layer_count)} "
            string += str(layer) + ",\n"
            layer_count += 1
        string = string[:-2]
        return string
