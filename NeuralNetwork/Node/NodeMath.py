import math

import sympy


def SigmoidFrom0To1(a) -> float:
    return 1/(1+math.exp(-a))


def SymbolSigmoid(a) -> float:
    return 1 / (1 + sympy.exp(-a))
