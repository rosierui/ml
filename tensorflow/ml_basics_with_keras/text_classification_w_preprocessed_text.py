# https://www.tensorflow.org/tutorials/keras/text_classification

# Movie reviews
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow import keras

# !pip install -q tensorflow-datasets
# conda install tensorflow-datasets, tensorflow-estimator, tensorflow-hub, tensorflow-metadata

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np

print(tf.__version__)

# Download the IMDB dataset
(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k', 
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure. 
    with_info=True)
    
# Try the encoder
# The dataset info includes the text encoder (a tfds.features.text.SubwordTextEncoder).
encoder = info.features['text'].encoder
print ('Vocabulary size: {}'.format(encoder.vocab_size))

# This text encoder will reversibly encode any string:
sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

assert original_string == sample_string

# The encoder encodes the string by breaking it into subwords or characters if
# the word is not in its dictionary. So the more a string resembles the dataset,
# the shorter the encoded representation will be.

for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))
  
# Explore the data
for train_example, train_label in train_data.take(1):
  print('Encoded text:', train_example[:10].numpy())
  print('Label:', train_label.numpy())

# The info structure contains the encoder/decoder. The encoder can be used to recover the original text:
encoder.decode(train_example)

# Prepare the data for training
# You will want to create batches of training data for your model. The reviews are
# all different lengths, so use padded_batch to zero pad the sequences while batching:

BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, train_data.output_shapes))

test_batches = (
    test_data
    .padded_batch(32, train_data.output_shapes))

# Each batch will have a shape of (batch_size, sequence_length) because the padding is dynamic each batch will have a different length:
for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)

"""
    Build the model
    The neural network is created by stacking layers—this requires two main architectural decisions:

    How many layers to use in the model?
    How many hidden units to use for each layer?
    In this example, the input data consists of an array of word-indices. The labels
    to predict are either 0 or 1. Let's build a "Continuous bag of words" style model for this problem:
"""
model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1, activation='sigmoid')])

model.summary()

# Loss function and optimizer
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
# Train the model by passing the Dataset object to the model's fit function. Set the number of epochs.
history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

# Evaluate the model
# And let's see how the model performs. Two values will be returned. Loss (a 
# number which represents our error, lower values are better), and accuracy.

loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

# Create a graph of accuracy and loss over time
# model.fit() returns a History object that contains a dictionary with everything that happened during training:

history_dict = history.history
history_dict.keys()

# There are four entries: one for each monitored metric during training and validation. 
# We can use these to plot the training and validation loss for comparison, as 
# well as the training and validation accuracy:

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()




  
  



