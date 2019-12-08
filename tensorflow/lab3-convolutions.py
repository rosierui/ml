# https://codelabs.developers.google.com/codelabs/tensorflow-lab3-convolutions/#0

import cv2
import numpy as np
from scipy import misc
img = misc.ascent()

import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(img)
plt.show()

