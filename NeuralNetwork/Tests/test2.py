from NeuralNetwork.NeuralNetwork import NeuralNetwork, LoadFromFile

nets = []
for _ in range(100):
    nets.append(NeuralNetwork(1, 1, 0, 0))

best_net = None
best_loss = 1

for index in range(len(nets)):
    nets[index] = nets[index].GetRandomizedBest([1], [1], 0.5, 100)
    loss = nets[index].GetLoss([1], [1])
    if loss < best_loss:
        best_net = nets[index]
        best_loss = loss
    if loss < 0.5:
        nets[index] = None

for _ in range(30000):
    best_net.CalculateSlopeValues([1], [1])
    best_net.MutateSlopeValues(1)

print(best_net.GetOutputs([1]))
print(best_net.GetOutputs([0]))
print(best_net)

# best_net.SaveToFile("Neural.pkl")
best_net = LoadFromFile("Neural.pkl")
print("after pickle")

print(best_net.GetOutputs([1]))
print(best_net.GetOutputs([0]))
print(best_net)
