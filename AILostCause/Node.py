import copy
import math
from random import random

from AILostCause.AIMath import SigmoidFrom0To1, PROPAGATION_STRENGTH, BIAS_CHANGE_COEFFICIENT


class Node:
    input_count = 1
    factors = []
    bias = 0

    def __init__(self, input_count, randomize=False):
        self.input_count = input_count
        self.factors = []
        self.bias = 0
        if randomize:
            self.Randomize()

    def GetOutput(self, input_list: list):
        if len(input_list) != self.input_count:
            return None
        result = 0
        for i in range(self.input_count):
            result += input_list[i] * self.factors[i]
        return SigmoidFrom0To1(result + self.bias)

    def GetRawOutput(self, input_list: list):
        if len(input_list) != self.input_count:
            return None
        result = 0
        for i in range(self.input_count):
            result += input_list[i] * self.factors[i]
        return result + self.bias

    def Randomize(self):
        self.factors = []
        for i in range(self.input_count):
            self.factors.append(1 - random() * 2)
        self.bias = 1 - random() * 2

    def CreateChild(self, variation: float):
        child = Node(input_count=self.input_count)
        child.factors = copy.deepcopy(self.factors)
        child.consts = copy.deepcopy(self.values)
        child.bias = self.bias
        child.Mutate(variation)
        return child

    def Mutate(self, variation: float):
        for i in range(self.input_count):
            self.factors[i] += variation * (1 - random() * 2)
        self.bias += variation * (1 - random() * 2)

    def BackpropagationDeviationList(self, input_list: list, output_deviation: float):

        back_strength = output_deviation * PROPAGATION_STRENGTH

        max_activation = max(input_list)
        min_activation = min(input_list)
        avg_activation = sum(input_list)/len(input_list)
        input_coefficient_list = \
            [(input_value-min_activation)/(max_activation-min_activation+1)+1 for input_value in input_list]

        factor_deviation_list = []
        for input_index in range(len(self.factors)):

            factor_value = self.factors[input_index]

            # if output_deviation > 0:
            #     # need to increase
            #     if input_list[input_index] > avg_activation:
            #         # increase output , input is high
            #         if factor_value > 0:
            #             # factor_value *= 1 + back_strength
            #             factor_value += back_strength*input_coefficient_list[input_index]
            #         elif factor_value < 0:
            #             # if negative, needs to move it to positive somehow
            #             factor_value = -factor_value
            #     elif input_list[input_index] < avg_activation:
            #         # increase output , input is low
            #         if factor_value > 0:
            #             # factor_value *= 1 + back_strength  # not sure, do i need to strengthen the connection when the input is 0?
            #             factor_value += back_strength*input_coefficient_list[input_index]
            #         elif factor_value < 0:
            #             # factor_value *= 1 + back_strength  # not sure , small decrease in effectiveness
            #             factor_value += back_strength*input_coefficient_list[input_index]
            # elif output_deviation < 0:
            #     # need to decrease
            #     if input_list[input_index] > avg_activation:
            #         # decrease output , input is high
            #         if factor_value > 0:
            #             # if positive, needs to move it to negative somehow
            #             factor_value = -factor_value
            #         elif factor_value < 0:
            #             # factor_value *= 1 + back_strength
            #             factor_value -= back_strength*input_coefficient_list[input_index]
            #     elif input_list[input_index] < avg_activation:
            #         # decrease output , input is low
            #         if factor_value > 0:
            #             # factor_value *= 1 - back_strength
            #             factor_value -= back_strength*input_coefficient_list[input_index]
            #         elif factor_value < 0:
            #             # factor_value *= 1 - back_strength
            #             factor_value -= back_strength*input_coefficient_list[input_index]

            factor_value += back_strength*input_coefficient_list[input_index]

            factor_deviation = factor_value - self.factors[input_index]
            factor_deviation_list.append(factor_deviation)
        return factor_deviation_list

    def DeviateWithFactorList(self, factor_deviation_list):
        max_deviation = max(factor_deviation_list)
        min_deviation = min(factor_deviation_list)

        # bias deviation calculations, adds to the bias if all deviations are positive and minimize the
        bias_change = None
        if max_deviation < 0:
            bias_change = max_deviation * BIAS_CHANGE_COEFFICIENT
        if min_deviation > 0:
            bias_change = min_deviation * BIAS_CHANGE_COEFFICIENT
        if type(bias_change) in (int, float):
            self.bias += bias_change * self.input_count
            factor_deviation_list = [factor_deviation - bias_change for factor_deviation in factor_deviation_list]

        for index in range(len(self.factors)):
            self.factors[index] += factor_deviation_list[index]

    def __str__(self):
        string = "("

        string += "factors["
        for factor in self.factors:
            string += "{:6.3f}, ".format(factor)

        string = string[:-2] + "], bias["
        string += "{:6.3f}, ".format(self.bias)

        string = string[:-2] + "]"
        string += ")"
        return string
