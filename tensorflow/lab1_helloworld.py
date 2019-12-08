# https://codelabs.developers.google.com/codelabs/tensorflow-lab1-helloworld/#0
# https://www.youtube.com/watch?v=KNAWp2S3w94 ML Zero to Hero, part 1

import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

"""
Next, the model uses the optimizer function to make another guess. Based on the 
loss function's result, it will try to minimize the loss. At this point maybe 
it will come up with something like `y=5x+5`. hile this is still pretty bad, 
it's closer to the correct result (i.e. the loss is lower).

stochastic gradient descent (sgd)
"""
model.compile(optimizer=tf.optimizers.SGD(), loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(f"model.predict([10.0]) - {model.predict([10.0])}")
print(f"model.predict([20.0]) - {model.predict([20.0])}")
