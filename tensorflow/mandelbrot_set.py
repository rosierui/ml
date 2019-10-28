# https://github.com/tensorflow/examples/blob/master/community/en/mandelbrot.ipynb

from __future__ import absolute_import, division, print_function, unicode_literals

# Import libraries for simulation
import tensorflow as tf
import numpy as np

# Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display

from tensorflow.python.framework.ops import Tensor, Graph

def DisplayFractal(a, fmt='jpeg'):
  """Display an array of iteration counts as a
     colorful picture of a fractal."""
  a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
  img = np.concatenate([10+20*np.cos(a_cyclic),
                        30+50*np.sin(a_cyclic),
                        155-80*np.cos(a_cyclic)], 2)
  img[a==a.max()] = 0
  a = img
  a = np.uint8(np.clip(a, 0, 255))
  f = BytesIO()
  PIL.Image.fromarray(a).save(f, fmt)
  display(Image(data=f.getvalue()))
  
# Use NumPy to create a 2D array of complex numbers

Y, X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y

xs = tf.constant(Z.astype(np.complex64))
zs = tf.Variable(xs)
not_diverged =tf.Variable(tf.zeros_like(xs, tf.bool))
ns = tf.Variable(tf.zeros_like(xs, tf.float32))

for i in range(200): 
  # Compute the new values of z: z^2 + x
  zs = zs*zs+xs
  # Have we diverged with this new value?
  not_diverged = tf.abs(zs) < 4
  # Operation to update the iteration count.
  #
  # Note: We keep computing zs after they diverge! This
  #       is very wasteful! There are better, if a little
  #       less simple, ways to do this.
  #
  ns = ns + tf.cast( not_diverged, tf.float32)
  
ns = ns.numpy()

DisplayFractal(ns)

