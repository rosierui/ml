# https://www.tensorflow.org/tutorials/keras/save_and_load

# Save and load models

#!pip install -q pyyaml h5py  # Required to save models in HDF5 format
#conda install pyyaml h5py

from __future__ import absolute_import, division, print_function, unicode_literals

import os

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

#Get an example dataset
#To demonstrate how to save and load weights, you'll use the MNIST dataset. To speed up these runs, use the first 1000 examples:

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

# Define a model
# Start by building a simple sequential model:

# Define a simple sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  return model

# Create a basic model instance
model = create_model()

# Display the model's architecture
model.summary()

"""
Save checkpoints during training
You can use a trained model without having to retrain it, or pick-up training where you left off—in case the training process was interrupted. The tf.keras.callbacks.ModelCheckpoint callback allows to continually save the model both during and at the end of training.

Checkpoint callback usage
Create a tf.keras.callbacks.ModelCheckpoint callback that saves weights only during training:
"""

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

#This creates a single collection of TensorFlow checkpoint files that are updated at the end of each epoch:

#!ls {checkpoint_dir}
"""
Create a new, untrained model. When restoring a model from weights-only, you must have a model with the same architecture as the original model. Since it's the same model architecture, you can share weights despite that it's a different instance of the model.

Now rebuild a fresh, untrained model, and evaluate it on the test set. An untrained model will perform at chance levels (~10% accuracy):
"""
# Create a basic model instance
model = create_model()

# Evaluate the model
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

# Then load the weights from the checkpoint and re-evaluate:

# Loads the weights
model.load_weights(checkpoint_path)

# Re-evaluate the model
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""
Checkpoint callback options
The callback provides several options to provide unique names for checkpoints and adjust the checkpointing frequency.

Train a new model, and save uniquely named checkpoints once every five epochs:
"""

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=5)

# Create a new model instance
model = create_model()

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
model.fit(train_images, 
              train_labels,
              epochs=50, 
              callbacks=[cp_callback],
              validation_data=(test_images,test_labels),
              verbose=0)

# Now, look at the resulting checkpoints and choose the latest one:

# !ls {checkpoint_dir}

latest = tf.train.latest_checkpoint(checkpoint_dir)
latest

# Note: the default tensorflow format only saves the 5 most recent checkpoints.
# To test, reset the model and load the latest checkpoint:

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

"""
What are these files?
The above code stores the weights to a collection of checkpoint-formatted files that contain only the trained weights in a binary format. Checkpoints contain: * One or more shards that contain your model's weights. * An index file that indicates which weights are stored in a which shard.

If you are only training a model on a single machine, you'll have one shard with the suffix: .data-00000-of-00001

Manually save weights
You saw how to load the weights into a model. Manually saving them is just as simple with the Model.save_weights method. By default, tf.keras—and save_weights in particular—uses the TensorFlow checkpoints format with a .ckpt extension (saving in HDF5 with a .h5 extension is covered in the Save and serialize models guide):
"""
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Create a new model instance
model = create_model()

# Restore the weights
model.load_weights('./checkpoints/my_checkpoint')

# Evaluate the model
loss,acc = model.evaluate(test_images,  test_labels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

# Save the entire model
# HDF5 format
# Keras provides a basic save format using the HDF5 standard.

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model to a HDF5 file.
# The '.h5' extension indicates that the model shuold be saved to HDF5.
model.save('my_model.h5') 

# Now, recreate the model from that file:

# Recreate the exact same model, including its weights and the optimizer
new_model = tf.keras.models.load_model('my_model.h5')

# Show the model architecture
new_model.summary()

# Check its accuracy:

loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

"""
This technique saves everything:

The weight values
The model's configuration(architecture)
The optimizer configuration
Keras saves models by inspecting the architecture. Currently, it is not able to save TensorFlow optimizers (from tf.train). When using those you will need to re-compile the model after loading, and you will lose the state of the optimizer.

SavedModel format
The SavedModel format is another way to serialize models. Models saved in this format can be restored using tf.keras.models.load_model and are compatible with TensorFlow Serving. The SavedModel guide goes into detail about how to serve/inspect the SavedModel. The section below illustrates the steps to saving and restoring the model.
"""

# Create and train a new model instance.
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save the entire model as a SavedModel.
# !mkdir -p saved_model
model.save('saved_model/my_model') 

"""
Instructions for updating:
If using Keras pass *_constraint arguments to layers.
INFO:tensorflow:Assets written to: saved_model/my_model/assets

The SavedModel format is a directory containing a protobuf binary and a Tensorflow checkpoint. Inspect the saved model directory:
"""
# my_model directory
# !ls saved_model

# Contains an assets folder, saved_model.pb, and variables folder.
# !ls saved_model/my_model

# Reload a fresh Keras model from the saved model:

new_model = tf.keras.models.load_model('saved_model/my_model')

# Check its architecture
new_model.summary()

# The restored model is compiled with the same arguments as the original model. Try running evaluate and predict with the loaded model:

# Evaluate the restored model
loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

print(new_model.predict(test_images).shape)

"""
Saving custom objects
If you are using the SavedModel format, you can skip this section. The key difference between HDF5 and SavedModel is that HDF5 uses object configs to save the model architecture, while SavedModel saves the execution graph. Thus, SavedModels are able to save custom objects like subclassed models and custom layers without requiring the orginal code.

To save custom objects to HDF5, you must do the following:

Define a get_config method in your object, and optionally a from_config classmethod.
get_config(self) returns a JSON-serializable dictionary of parameters needed to recreate the object.
from_config(cls, config) uses the returned config from get_config to create a new object. By default, this function will use the config as initialization kwargs (return cls(**config)).
Pass the object to the custom_objects argument when loading the model. The argument must be a dictionary mapping the string class name to the Python class. E.g. tf.keras.models.load_model(path, custom_objects={'CustomLayer': CustomLayer})
See the Writing layers and models from scratch tutorial for examples of custom objects and get_config.
"""
