# https://colab.research.google.com/notebooks/mlcc/tensorflow_programming_concepts.ipynb#scrollTo=SDbi6heigEGA  

from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt # Dataset visualization.
import numpy as np              # Low-level numerical Python library.
import pandas as pd             # Higher-level numerical Python library.


# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
  # Assemble a graph consisting of the following three operations:
  #   * Two tf.constant operations to create the operands.
  #   * One tf.add operation to add the two operands.
  x = tf.constant(8, name="x_const")
  y = tf.constant(5, name="y_const")
  my_sum = tf.add(x, y, name="x_y_sum")


  # Now create a session.
  # The session will run the default graph.
  with tf.Session() as sess:
    print(my_sum.eval())

  z = tf.constant(4, name="z_const")
  new_sum = tf.add(my_sum, z, name="x_y_z_sum")

  with tf.Session() as sess:
    print(new_sum.eval())
  
