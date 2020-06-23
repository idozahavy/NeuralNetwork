import numpy as np


# np.random.seed(0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# Input datasets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
expected_output = np.array([[0], [1], [1], [0]])

epochs = 1
lr = 0.1
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons, hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1, hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons, outputLayerNeurons))
output_bias = np.random.uniform(size=(1, outputLayerNeurons))

hidden_weights = np.array([[0.8, 0.4], [0.1, 0.05]])
hidden_bias = np.array([[0.2, 0.6]])
output_weights = np.array([[0.35], [0.21]])
output_bias = np.array([[0.7]])

# print("Initial hidden weights: ", end='')
# print(*hidden_weights)
# print("Initial hidden biases: ", end='')
# print(*hidden_bias)
# print("Initial output weights: ", end='')
# print(*output_weights)
# print("Initial output biases: ", end='')
# print(*output_bias)

# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs, hidden_weights)  # correct on mine
    hidden_layer_activation += hidden_bias  # correct on mine
    hidden_layer_output = sigmoid(hidden_layer_activation)  # correct on mine

    output_layer_activation = np.dot(hidden_layer_output, output_weights)  # correct on mine
    output_layer_activation += output_bias  # correct on mine
    predicted_output = sigmoid(output_layer_activation)  # correct on mine

    # Backpropagation
    error = expected_output - predicted_output  # correct on mine
    d_predicted_output = error * sigmoid_derivative(predicted_output)  # correct on mine = node.cost_value_der

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    print(f"output weights = {output_weights.T}")
    print(f"error_hidden_layer = {error_hidden_layer}")
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    print(f"d_hidden_layer = {d_hidden_layer}")

    print(f"outputs bias changes - {np.sum(d_predicted_output, axis=0, keepdims=True)}")
    print(f"hidden bias changes - {np.sum(d_hidden_layer, axis=0, keepdims=True)}")

    print(f"output weights changes - {hidden_layer_output.T.dot(d_predicted_output)}")
    print(f"hidden weights changes - {inputs.T.dot(d_hidden_layer)}")

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output, axis=0, keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr


# print("Final hidden weights: ", end='')
# print(*hidden_weights)
# print("Final hidden bias: ", end='')
# print(*hidden_bias)
# print("Final output weights: ", end='')
# print(*output_weights)
# print("Final output bias: ", end='')
# print(*output_bias)

print("\nOutput from neural network after 10,000 epochs: ", end='')
print(*predicted_output)
