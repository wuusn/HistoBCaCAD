from .multiRun import *
from .wsi2patch import *
import numpy as np

def majorityVoting(a):
    # return the most common label
    a = np.array(a)
    a = a[::-1]
    res = np.argmax(a)
    res = abs(2-res)
    return res

def maxProbability(a):
    # return the pred label with max probability
    pass

def maxPredict(a):
    # return the max pred label
    l = len(a)
    for i in range(l):
        if a[l-i-1]!=0:
            return l-i-1

def sumOfProbability(a):
    # return the the label with max sumed probability
    pass
