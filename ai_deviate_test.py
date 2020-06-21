import copy

from AILostCause.AIMath import MatricesAdd
from AILostCause.AI import AI

ai_original = AI(2, 2, nodes_layers=(2, 0))
ai = copy.deepcopy(ai_original)

print(ai)

input_output_matrix = [
    {"input": [1, 1], "output": [1, 1]},
    {"input": [1, 0], "output": [1, 0]},
    {"input": [0, 1], "output": [0, 1]},
    {"input": [0, 0], "output": [0, 0]},
    {"input": [0.5, 0], "output": [0.5, 0]},
    {"input": [0, 0.2], "output": [0, 0.2]},
    {"input": [1, 0.7], "output": [1, 0.7]},
    {"input": [0.1, 0.6], "output": [0.1, 0.6]},
]

for _ in range(1000):
    print()
    print("new Propagation")
    print()
    back_matrix = None
    for input_output in input_output_matrix:
        current_backpro = ai.BackpropagationDeviationMatrixList(input_output["input"], input_output["output"])
        # print(f"adds {current_backpro}")
        if back_matrix is None:
            back_matrix = current_backpro
        else:
            back_matrix = MatricesAdd(back_matrix, current_backpro)
        print(f"     {back_matrix}")
    print(back_matrix)
    print("after propagation")
    ai.DeviateWithMatrixList(back_matrix)
    print(ai)
    print(f"[1, 1] result - {ai.GetOutput([1, 1])}")

print()
print(f"[1, 1] original result - {ai_original.GetOutput([1, 1])}")
print(f"[1, 1] result - {ai.GetOutput([1, 1])}")
print(f"[1, 0] result - {ai.GetOutput([1, 0])}")
print(f"[0, 1] result - {ai.GetOutput([0, 1])}")
print(f"[0, 0] result - {ai.GetOutput([0, 0])}")
