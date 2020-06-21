import copy

from AILostCause.Node import Node

nd_original = Node(2, randomize=True)
nd = copy.deepcopy(nd_original)

print(nd)
print(f"[1, 1] result - {nd.GetOutput([1, 1])}")

correct_output = 0.5

for _ in range(50):
    print()
    print("new Propagation")
    print()
    output_deviation = correct_output - nd.GetOutput([1, 1])
    back_list = nd.BackpropagationDeviationList([1, 1], output_deviation)
    print(back_list)
    print("after propagation")
    nd.DeviateWithFactorList(back_list)
    print(nd)
    print(f"[1, 1] result - {nd.GetOutput([1, 1])}")

print()
print(f"[1, 1] original result - {nd_original.GetOutput([1, 1])}")
print(f"[1, 1] result - {nd.GetOutput([1, 1])}")
