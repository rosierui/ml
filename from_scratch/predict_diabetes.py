# https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/

"""
Neural Network Theory
A neural network is a supervised learning algorithm which means that we provide it the input data containing the independent 
variables and the output data that contains the dependent variable. 

In the beginning, the neural network makes some random predictions, these predictions are matched with the correct output and 
the error or the difference between the predicted values and the actual values is calculated. The function that finds the 
difference between the actual value and the propagated values is called the cost function. The cost here refers to the error. 
Our objective is to minimize the cost function. Training a neural network basically refers to minimizing the cost function. 

Back Propagation
Step 1: Calculating the cost
The first step in the back propagation section is to find the "cost" of the predictions. The cost of the prediction can simply 
be calculated by finding the difference between the predicted output and the actual output. The higher the difference, the 
higher the cost will be.

Step 2: Minimizing the cost
Our ultimate purpose is to fine-tune the knobs of our neural network in such a way that the cost is minimized. If your look 
at our neural network, you'll notice that we can only control the weights and the bias. Everything else is beyond our control. 
We cannot control the inputs, we cannot control the dot products, and we cannot manipulate the sigmoid function.

In order to minimize the cost, we need to find the weight and bias values for which the cost function returns the smallest 
value possible. The smaller the cost, the more correct our predictions are.

This is an optimization problem where we have to find the function minima.
To find the minima of a function, we can use the gradient decent algorithm.
"""

import numpy as np
feature_set = np.array([[0,1,0],[0,0,1],[1,0,0],[1,1,0],[1,1,1]])
labels = np.array([[1,0,0,1,1]])
labels = labels.reshape(5,1)

np.random.seed(42)
weights = np.random.rand(3,1)
bias = np.random.rand(1)
lr = 0.05

def sigmoid(x):
    return 1/(1+np.exp(-x))

#And the method that calculates the derivative of the sigmoid function is defined as follows:
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))

for epoch in range(20000):
    inputs = feature_set

    # feedforward step1
    XW = np.dot(feature_set, weights) + bias

    #feedforward step2
    z = sigmoid(XW)


    # backpropagation step 1
    error = z - labels

    print(error.sum())

    # backpropagation step 2
    dcost_dpred = error
    dpred_dz = sigmoid_der(z)

    z_delta = dcost_dpred * dpred_dz

    inputs = feature_set.T
    weights -= lr * np.dot(inputs, z_delta)

    for num in z_delta:
        bias -= lr * num

XW = np.dot(feature_set, weights) + bias
z = sigmoid(XW)
error = z - labels

dcost_dpred = error # ........ (2)
dpred_dz = sigmoid_der(z) # ......... (3)

#slope = input x dcost_dpred x dpred_dz
#Take a look at the following three lines:
z_delta = dcost_dpred * dpred_dz
inputs = feature_set.T
weights -= lr * np.dot(inputs, z_delta)

"""
You can now try and predict the value of a single instance. Let's suppose we have a record of a patient that comes in who smokes, 
is not obese, and doesn't exercise. Let's find if he is likely to be diabetic or not. The input feature will look like this: [1,0,0].
"""
single_point = np.array([1,0,0])
result = sigmoid(np.dot(single_point, weights) + bias)
print("")
print(result)
