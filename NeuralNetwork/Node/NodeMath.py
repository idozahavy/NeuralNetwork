import math

import sympy
import numpy


def Sigmoid(a) -> float:
    if a > 230:
        return 1
    if a < -230:
        return 0
    return 1/(1+math.exp(-a))


def SymbolSigmoid(a) -> float:
    return 1 / (1 + sympy.exp(-a))
