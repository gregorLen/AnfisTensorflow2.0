# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:12:52 2020

@author: Gregor
"""
import math
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x, gamma=1, c=0):
    return 1 / (1 + np.exp(-gamma*(x-c)))
xn = np.arange(-4,4,0.1)

for gamma in [2,-2]:
    plt.plot(xn, sigmoid(xn, gamma=gamma))
