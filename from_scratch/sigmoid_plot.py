# https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/

import numpy as np

input = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1/(1+np.exp(-x))

from matplotlib import pyplot as plt
plt.plot(input, sigmoid(input), c="r")
