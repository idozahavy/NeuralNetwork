import time

from AILostCause.BetterNode import Node

starting_nodes = [Node() for _ in range(784)]
for strt in starting_nodes:
    strt.SetActivation(1)
hidden1 = [Node(input_nodes=starting_nodes) for _ in range(16)]
hidden2 = [Node(input_nodes=hidden1) for _ in range(16)]
output_nodes = [Node(input_nodes=hidden2) for _ in range(10)]

start_time = time.time()
for _ in range(1):
    for output in output_nodes:
        a = output.CalcActivation()
print(time.time() - start_time)

exit()


start1 = Node()
start1.SetActivation(0)
start2 = Node()
start2.SetActivation(0)

n11 = Node(input_nodes=[start1, start2])
n12 = Node(input_nodes=[start1, start2])
n13 = Node(input_nodes=[start1, start2])
n14 = Node(input_nodes=[start1, start2])

n21 = Node(input_nodes=[n11, n12, n13, n14])
n22 = Node(input_nodes=[n11, n12, n13, n14])
n23 = Node(input_nodes=[n11, n12, n13, n14])
n24 = Node(input_nodes=[n11, n12, n13, n14])


output1 = Node(input_nodes=[n21, n22, n23, n24])
output2 = Node(input_nodes=[n21, n22, n23, n24])
print()

input_output_list = [
    {
        "input": [1, 0],
        "output": [0, 1]
    },
    {
        "input": [1, 1],
        "output": [1, 0]
    },
    {
        "input": [0, 1],
        "output": [0, 1]
    },
    {
        "input": [0, 0],
        "output": [1, 0]
    }
]

for input_output in input_output_list:
    start1.SetActivation(input_output["input"][0])
    start2.SetActivation(input_output["input"][1])
    print(
        f"input = [{input_output['input'][0]},{input_output['input'][1]}], output = {output2.CalcActivation(recalculate=True)}")

# exit()
output1_total = 0
output2_total = 0
output1_add_total = 0
output2_add_total = 0

for _ in range(500):
    for input_output in input_output_list:
        start1.SetActivation(input_output["input"][0])
        start2.SetActivation(input_output["input"][1])
        # if input_output is input_output_list[3]:
        #     print("a")

        output1_time = time.time()
        output1_deviations = output1.GetBackpropagationDeviationNode(input_output["output"][0])
        output1_total += time.time() - output1_time

        output2_time = time.time()
        output2_deviations = output2.GetBackpropagationDeviationNode(input_output["output"][1])
        output2_total += time.time() - output2_time

        output1_time = time.time()
        output1_deviations.AddDeviations(output1)
        output1_add_total += time.time() - output1_time

        output2_time = time.time()
        output2_deviations.AddDeviations(output2)
        output2_add_total += time.time() - output2_time

        output1 = output1_deviations  # output1  and output2 are separated need to make them refer the original nodes
        output2 = output2_deviations
        # print("After Propagation " + str(output2))
    output1.Mutate()
    output2.Mutate()
    print(f"output activation - {output2.CalcActivation()}")
    # print(output2)
    # print()
print()
print()
for input_output in input_output_list:
    start1.SetActivation(input_output["input"][0])
    start2.SetActivation(input_output["input"][1])
    print(
        f"input = [{input_output['input'][0]},{input_output['input'][1]}], output = {output2.CalcActivation(recalculate=True)}")

print(f"time passes = {time.time() - start_time} seconds")
print()
print(f"output1 total time = {output1_total} seconds")
print(f"output2 total time = {output2_total} seconds")
print()
print(f"output1 add total time = {output1_add_total} seconds")
print(f"output2 add total time = {output2_add_total} seconds")
