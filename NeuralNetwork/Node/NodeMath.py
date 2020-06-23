import math


def Sigmoid(a) -> float:
    if a > 230:
        return 1
    if a < -230:
        return 0
    return 1/(1+math.exp(-a))


def SigmoidDerivative(value):  # 0.0000008 secs once
    return Sigmoid(value)*(1-Sigmoid(value))
