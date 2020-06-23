from NeuralNetwork.NeuralNetwork import NeuralNetwork
from NeuralNetwork.Node.BasicNodeClasses import InputLinkedNode, NodeLink

net = NeuralNetwork(2, 1, 1, 2)

# hidden_bias = np.array([[0.2, 0.6]])
net.hidden_nodes[0].bias = 0.2
net.hidden_nodes[1].bias = 0.6

# hidden_weights = np.array([[0.8, 0.4], [0.1, 0.05]])
net.hidden_nodes[0].input_links[0].factor = 0.8
net.hidden_nodes[0].input_links[1].factor = 0.1
net.hidden_nodes[1].input_links[0].factor = 0.4
net.hidden_nodes[1].input_links[1].factor = 0.05

# output_bias = np.array([[0.7]])
net.output_nodes[0].bias = 0.7

# output_weights = np.array([[0.35], [0.21]])
net.output_nodes[0].input_links[0].factor = 0.35
net.output_nodes[0].input_links[1].factor = 0.21


inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
expected_output = [[0], [1], [1], [0]]

for index in range(4):
    output = net.GetOutputs(inputs[index])
    print(output)
    net.CalculateSlopeValues(inputs[index], expected_output[index])

for node in net.node_dictionary.values():
    print(node, end="")
    if isinstance(node, InputLinkedNode):
        print(f"bias_change = {node.cost_bias_der}")
        print("----links----")
        for link in node.input_links:
            link: NodeLink
            print(f"factor change = {link.cost_factor_der}")
    print()




