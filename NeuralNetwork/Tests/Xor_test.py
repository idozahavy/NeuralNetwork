import time
from random import random

from NeuralNetwork.NeuralNetwork import NeuralNetwork

net = NeuralNetwork(2, 1, 1, 2)

inputs_outputs = [
    {"input": [1, 1], "output": [0]},
    {"input": [1, 0], "output": [1]},
    {"input": [0, 1], "output": [1]},
    {"input": [0, 0], "output": [0]},
    # {"input": [5, 3], "output": [1, 0]},
    # {"input": [9, 0], "output": [0, 1]},
    # {"input": [0, 2], "output": [0, 1]},
    # {"input": [-0.2, 0], "output": [1, 0]},
    # {"input": [0.7, -0.3], "output": [0, 1]}
]
for input_output in inputs_outputs:
    print(f"input = {input_output['input']}  , output = {net.GetOutputs(input_output['input'])}")
print()
timer = time.time()
for _ in range(16000):
    loss = 0
    for input_output in inputs_outputs:
        net.CalculateCosts(input_output["input"], input_output["output"])
        loss += net.GetLoss(input_output["input"], input_output["output"])
    # for _ in range(30):
    #     a = int(random()*2)
    #     b = int(random()*2)
    #     net.CalculateCosts([a, b], [int(a > b), int(b > a)])
    net.MutateCosts(coefficient=0.01)
    print(loss)
    # for input_output in inputs_outputs:
    #     print(f"input = {input_output['input']}  , output = {net.GetOutputs(input_output['input'])}")
    # print()
print(time.time() - timer)
print()
for input_output in inputs_outputs:
    print(f"input = {input_output['input']}  , output = {net.GetOutputs(input_output['input'])}")
print()


