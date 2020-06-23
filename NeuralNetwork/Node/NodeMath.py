import math


def Sigmoid(a) -> float:
    if a > 230:
        return 1
    if a < -230:
        return 0
    return 1 / (1 + math.exp(-a))


def SigmoidDerivative(value):
    return value * (1 - value)  # works
    # return Sigmoid(value) * (1 - Sigmoid(value))
