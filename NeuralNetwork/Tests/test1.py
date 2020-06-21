import time

import sympy

from NeuralNetwork.Node.BasicNodeClasses import NodeLink
from NeuralNetwork.Node.HiddenNode import HiddenNode
from NeuralNetwork.Node.InputNode import InputNode
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

timer = time.time()
for _ in range(1_000_000):
    # a = output1.GetActivation()  # 1_000_000 times = 3.9 seconds (math instead of sympy(80 seconds))
    a = output1.GetActivation(save=True)  # 1_000_000 times = 0.46 seconds
print(time.time()-timer)


