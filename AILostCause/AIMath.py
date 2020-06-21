import math
import numpy
import sympy

PROPAGATION_STRENGTH = 0.1
BIAS_CHANGE_COEFFICIENT = 0.2
LAYER_BACKPROPAGATION_COEFFICIENT = 1


def SigmoidFrom0To1(a) -> float:
    return 1/(1+sympy.exp(-a))


def CalculateCost(actual_output, desired_output):
    return math.pow(actual_output-desired_output, 2)


def MatrixSumLowestList(matrix):
    if type(matrix) in (float, int, numpy.float64):
        return matrix
    if type(matrix) is list:
        if len(matrix) < 1:
            return 0
        if type(matrix[0]) is list:
            result = []
            for ls in matrix:
                result.append(MatrixSumLowestList(ls))
            return result
        elif type(matrix[0]) in (float, int, numpy.float64):
                sum = 0
                result = []
                for item in matrix:
                    if type(item) in (float, int, numpy.float64):
                        sum += item
                    else:
                        result.append(MatrixSumLowestList(item))
                if not result:
                    return sum
                result.insert(0, sum)
                return result
    return None


def MatricesAdd(matrix1, matrix2):
    result = []
    if type(matrix1) is type(matrix2):
        if type(matrix1) is list:
            for index in range(len(matrix1)):
                result.append(MatricesAdd(matrix1[index], matrix2[index]))
        elif type(matrix1) in (float, int, numpy.float64):
            return matrix1 + matrix2
    return result


def MatricesMinus(matrix1, matrix2):
    result = []
    if type(matrix1) is type(matrix2):
        if type(matrix1) is list:
            for index in range(len(matrix1)):
                result.append(MatricesMinus(matrix1[index], matrix2[index]))
        elif type(matrix1) in (float, int, numpy.float64):
            return float(matrix1 - matrix2)
    elif type(matrix1) in (float, int, numpy.float64) and type(matrix2) in (float, int, numpy.float64):
        return MatricesMinus(float(matrix1),float(matrix2))
    return result


def MatrixDivide(matrix, divider):
    if type(matrix) is list:
        for index in range(len(matrix)):
            matrix[index] = MatrixDivide(matrix[index], divider)
    elif type(matrix) in (float, int):
        return matrix / divider
    return matrix
