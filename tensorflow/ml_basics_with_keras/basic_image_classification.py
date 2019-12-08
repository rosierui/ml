# https://www.tensorflow.org/tutorials/keras/classification
# Computer Vision Example: https://goo.gle/34cHkDk
# https://www.youtube.com/watch?v=bemDFpNooA8&vl=en ML Zero to Hero, part 2

# https://codelabs.developers.google.com/codelabs/tensorflow-lab2-computervision/#0 *
# https://colab.research.google.com/github/lmoroney/mlday-tokyo/blob/master/Lab2-Computer-Vision.ipynb#scrollTo=6tki-Aro_Uax 
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

print("Import the Fashion MNIST dataset")
# beginner.py - mnist = tf.keras.datasets.mnist
fashion_mnist = keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


print("Explore the data")
print(f"training_images.shape: {training_images.shape}")
print(f"len(training_labels): {len(training_labels)}")

# Each label is an integer between 0 and 9:
print(f"training_labels: {training_labels}")

# There are 10,000 images in the test set. Again, each image is represented as 28 x 28 pixels:
print(f"test_images.shape : {test_images.shape}")
print(f"len(test_labels) : {len(test_labels)}")
print(f"test_labels: {test_labels}")


print("\n1 - plt.imshow(training_images[0])")
plt.imshow(training_images[0])

print("\n2 - training_images[0]")
print(training_images[0])

print("\n3 - training_labels[0]")
print(training_labels[0])

"""
Preprocess the data
The data must be preprocessed before training the network. 
If you inspect the first image in the training set, you will see that the pixel values fall in the range of 0 to 255:

The imshow function displays the value low (and any value less than low ) as black, and it displays the value high 
(and any value greater than high ) as white. Values between low and high are displayed as intermediate shades of gray, 
using the default number of gray levels
"""
plt.figure()
plt.imshow(training_images[0])
plt.colorbar()
plt.grid(False)
plt.xlabel(class_names[training_labels[0]])
plt.ylabel(training_labels[0])
plt.show()

plt.figure()
plt.imshow(training_images[1])
#plt.colorbar()
plt.grid(True)
plt.xticks([])
plt.yticks([])
plt.xlabel(class_names[training_labels[1]])
plt.ylabel(training_labels[1])
plt.show()

"""
Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
To do so, divide the values by 255. It's important that the training set and the testing set be preprocessed in the same way:
"""

training_images = training_images / 255.0
test_images = test_images / 255.0

"""
To verify that the data is in the correct format and that you're ready to build and train the network, 
let's display the first 25 images from the training set and display the class name below each image.
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i]])
plt.show()

# Building the neural network requires configuring the layers of the model, then compiling the model.
model = keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Rectified Linear Units (ReLU)
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)   # 0 ~ 9
])

"""
The `metrics=` parameter, this allows TensorFlow to report back about how accurate the training is against the 
test set. It measures how many it got right and wrong, and reports back on how it's doing.
"""
model.compile(optimizer=tf.optimizers.Adam(), # https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/optimizers
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
"""
https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
batch size = the number of training examples in one forward/backward pass. number of iterations = number of passes, 
each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we 
do not count the forward pass and backward pass as two different passes).
"""
model.fit(training_images, training_labels, epochs=5)

"""
Test the model
Next, compare how the model performs on the test dataset:
"""
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)

"""
Make predictions
"""
predictions = model.predict(test_images)
print(predictions[0])
np.argmax(predictions[0])
test_labels[0]

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])

#@title MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.  