import time
from random import random

from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.Node.BasicNodeClasses import NodeLink
from NeuralNetwork.Node.HiddenNode import HiddenNode
from NeuralNetwork.Node.InputNode import InputNode
from NeuralNetwork.Node.NodeMath import SigmoidDerivative
from NeuralNetwork.Node.OutputNode import OutputNode

input1 = InputNode(activation=0, node_number=0)
hidden1 = HiddenNode(node_number=1, node_layer=0)
output1 = OutputNode(node_number=2)

print(input1)
print(hidden1)
print(output1)

hidden1_link1 = NodeLink(input1, hidden1)
output1_link1 = NodeLink(hidden1, output1)

hidden1.input_links.append(hidden1_link1)
output1.input_links.append(output1_link1)

# timer = time.time()
# for _ in range(1_000_000):
#     # a = output1.GetActivation()  # 1_000_000 times = 3.9 seconds (math instead of sympy(80 seconds))
#     a = output1.GetActivation(save=True)  # 1_000_000 times = 0.46 seconds
# print(time.time()-timer)

net = NeuralNetwork(input_count=784, output_count=10, hidden_layer_count=2, hidden_layer_nodes_count=16)

inputs = [0 for i in range(784)]

timer = time.time()
for _ in range(1):
    for _ in range(100):
        # net = NeuralNetwork(input_count=100, output_count=100, hidden_layer_count=100, hidden_layer_nodes_count=100)
        # print(net)

        # a = net.GetOutputs(inputs)
        # print(a)

        # h = net.GetHiddenNode(0, 1)  # 0
        # print(h)

        net.CalculateCosts(inputs, [1 for i in range(10)])

        # a = SigmoidDerivative(inputs.pop())
        pass
    net.MutateCosts()
    print("Mutated")
print(time.time()-timer)

print()

print(net.GetOutputs([-50 for _ in range(784)]))

for _ in range(100):
    a = int(random() * 2)
    b = int(random() * 2)
    xor = a ^ b
    print(f"{[a, b]}, {[int(xor == 0), xor]}")
