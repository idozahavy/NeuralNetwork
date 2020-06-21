import math
import sys

from AILostCause.AI import AI
from AILostCause.AIMath import MatricesAdd, MatrixDivide


def GetScore(calc_result, real_result):
    diff = math.fabs(real_result-calc_result)
    score = 1.0-diff
    return score


ai = AI(2, 2, nodes_layers=(2, 2))
print(ai)
print("input = [1, 1]")
print("output = "+str(ai.GetOutput([1, 1])))
deviation = ai.BackPropagationDeviationMatrix([1, 1], [0, 0])
print(f"deviation values = {deviation}")
ai.DeviateWithMatrixList(deviation)
print("After Deviation fix")
print(ai)

mat = MatricesAdd(deviation, deviation)
mat = MatrixDivide(mat, 2)
print(mat)

ai_list = []
for _ in range(100):
    ai = AI(2, 1, nodes_layers=(2, 1))
    ai_list.append(ai)

input_list1 = [1, 1]
input_list2 = [1, 0]
input_list3 = [0, 1]
input_list4 = [0, 0]

output1 = 0
output2 = 1
output3 = 1
output4 = 0

best_index = None
best_score = -sys.float_info.max
total_score = 0

for _ in range(100):

    mid_score = total_score / len(ai_list)
    for i in range(len(ai_list)):
        if ai_list[i].score < mid_score:
            if best_index:
                ai_list[i] = ai_list[best_index].CreateChild(0.3)
            else:
                ai_list[i].Mutate(1)

    best_index = None
    best_score = -sys.float_info.max
    total_score = 0

    for i in range(len(ai_list)):
        ai = ai_list[i]
        ai_score = 0
        ai_score += GetScore(ai.GetOutput(input_list1)[0], output1)
        ai_score += GetScore(ai.GetOutput(input_list2)[0], output2)
        ai_score += GetScore(ai.GetOutput(input_list3)[0], output3)
        ai_score += GetScore(ai.GetOutput(input_list4)[0], output4)
        ai_list[i].score = ai_score
        total_score += ai_score
        if ai_score > best_score:
            best_index = i
            best_score = ai_score

print(f"best score is ={best_score}")
print(ai_list[best_index])

print(ai_list[best_index].GetOutput([1, 1]))
print(ai_list[best_index].GetOutput([0, 1]))
print(ai_list[best_index].GetOutput([1, 0]))
print(ai_list[best_index].GetOutput([0, 0]))

output1 = [1, 0]
output2 = [0, 1]
output3 = [0, 1]
output4 = [1, 0]

output11 = 0
output01 = 0
output10 = 0
output00 = 0

ai_list = []
for _ in range(1):
    ai = AI(2, 2, nodes_layers=(2, 1))
    ai_list.append(ai)

for _ in range(100):
    for i in range(len(ai_list)):
        ai = ai_list[i]

        dif_output11 = ai.GetOutput([1, 1])[1] - output11
        dif_output01 = ai.GetOutput([0, 1])[1] - output01
        dif_output10 = ai.GetOutput([1, 0])[1] - output10
        dif_output00 = ai.GetOutput([0, 0])[1] - output00

        output11 = ai.GetOutput([1, 1])[1]
        output01 = ai.GetOutput([0, 1])[1]
        output10 = ai.GetOutput([1, 0])[1]
        output00 = ai.GetOutput([0, 0])[1]

        deviation = ai.BackPropagationDeviationMatrix(input_list1, output1)
        for _ in range(10):
            deviation = MatricesAdd(ai.BackPropagationDeviationMatrix(input_list1, output1), deviation)
            deviation = MatricesAdd(ai.BackPropagationDeviationMatrix(input_list2, output2), deviation)
            deviation = MatricesAdd(ai.BackPropagationDeviationMatrix(input_list3, output3), deviation)
            deviation = MatricesAdd(ai.BackPropagationDeviationMatrix(input_list4, output4), deviation)
        # deviation = MatrixDivide(deviation, 10)
        deviation = MatrixDivide(deviation, 10)
        ai.DeviateWithMatrixList(deviation)

print(ai_list[0].GetOutput([1, 1]))
print(ai_list[0].GetOutput([0, 1]))
print(ai_list[0].GetOutput([1, 0]))
print(ai_list[0].GetOutput([0, 0]))
