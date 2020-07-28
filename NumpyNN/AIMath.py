import functools
import math
import numpy as np
from typing import Union

from scipy import special

numberTypes = (float, np.float64, np.float32, int, np.int32, np.int64)
onesDict = {}


def ones(value: np.ndarray):
    if value.shape not in onesDict.keys():
        newOnes = np.ones(value.shape)
        onesDict[value.shape] = newOnes
        return newOnes
    return onesDict[value.shape]


def Sigmoid(value: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    if isinstance(value, numberTypes):
        if value > 230:
            return 1
        elif value < -230:
            return 0
        return 1 / (1 + math.exp(-value))
    elif isinstance(value, (np.ndarray,)):
        return ones(value) / (ones(value) + np.exp(-value))
    else:
        assert value is None


def SigmoidDerivative(value: Union[np.ndarray, float]):
    if isinstance(value, numberTypes):
        return value * (1 - value)
    elif isinstance(value, np.ndarray):
        return value * (ones(value) - value)
    else:
        assert value is None


def Softplus(value: Union[np.ndarray, float]):
    if isinstance(value, numberTypes):
        return math.log(1 + math.exp(value))
    elif isinstance(value, np.ndarray):
        return np.log(ones(value) + np.exp(value))
    else:
        assert value is None


def SoftplusDerivative(value: Union[np.ndarray, float]):
    if isinstance(value, numberTypes):
        return value / (1 + value)
    elif isinstance(value, np.ndarray):
        return value / (ones(value) + value)
    else:
        assert value is None


def Softmax(value: np.ndarray):
    if isinstance(value, np.ndarray):
        exp = np.exp(value - (ones(value) * np.max(value)))
        return exp / np.sum(exp)
    else:
        assert value is None


def SoftmaxDerivative(value: np.ndarray):
    if isinstance(value, np.ndarray):
        return value * (ones(value) - value)
        # if need derivative of same value, because every element changes the other elements value by definition
    else:
        assert value is None


def Relu(value: Union[np.ndarray, float]):
    if isinstance(value, numberTypes):
        return max(0, value)
    elif isinstance(value, np.ndarray):
        return np.maximum(ones(value)*0, value)
    else:
        assert value is None


def ReluDerivative(value: Union[np.ndarray, float]):
    if isinstance(value, numberTypes):
        if value > 0:
            return 1
        else:
            return 0
    elif isinstance(value, np.ndarray):
        return np.where(value > 0, 1, 0)
    else:
        assert value is None


# Todo maybe add leaky-relu = max(value,0) + x*min(value,0)
