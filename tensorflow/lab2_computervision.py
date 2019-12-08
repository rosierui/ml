# https://codelabs.developers.google.com/codelabs/tensorflow-lab2-computervision/#3

import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
#training_images=training_images/255.0
#test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),  # 512 functions w/ randomly initialized variables
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # 10 outputs
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')  # initialize the functions

model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
classifications = model.predict(test_images)

print(f"len(classifications): {len(classifications)}")
print(classifications[0],classifications[0] )

print(f"test_labels: {len(test_labels)}")
print(test_labels[0], test_labels[1])
