import time

from NeuralNetwork.NeuralNetwork import NeuralNetwork

net = NeuralNetwork(2, 1, 1, 2)

inputs_outputs = [
    {"input": [1, 1], "output": [0]},
    {"input": [1, 0], "output": [1]},
    {"input": [0, 1], "output": [1]},
    {"input": [0, 0], "output": [0]}
]
for input_output in inputs_outputs:
    print(f"input = {input_output['input']}  , output = {net.GetOutputs(input_output['input'])}")
print()

last_loss = 1

timer = time.time()
for _ in range(8000):
    loss = 0
    for input_output in inputs_outputs:
        net.CalculateSlopeValues(input_output["input"], input_output["output"])
        loss += net.GetLoss(input_output["input"], input_output["output"])
    loss_change = loss - last_loss
    net.MutateSlopeValues(coefficient=loss*2)
    print(loss)
    last_loss = loss
print(time.time() - timer)
print()
for input_output in inputs_outputs:
    print(f"input = {input_output['input']}  , output = {net.GetOutputs(input_output['input'])}")
print()


