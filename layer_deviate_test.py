import copy

from AILostCause.AIMath import MatricesMinus
from AILostCause.Layer import Layer
from AILostCause.Node import Node

layer_original = Layer(2, 2)
ly = copy.deepcopy(layer_original)

print(ly)
print(f"[1, 1] result - {ly.GetOutput([1, 1])}")

correct_output = [0, 1]

for _ in range(100):
    print()
    print("new Propagation")
    print()
    output_deviation = MatricesMinus(correct_output, ly.GetOutput([1, 1]))
    back_matrix = ly.BackpropagationDeviationMatrix([1, 1], output_deviation)
    print(back_matrix)
    print("after propagation")
    ly.DeviateWithMatrix(back_matrix)
    print(ly)
    print(f"[1, 1] result - {ly.GetOutput([1, 1])}")

print()
print(f"[1, 1] original result - {layer_original.GetOutput([1, 1])}")
print(f"[1, 1] result - {ly.GetOutput([1, 1])}")
