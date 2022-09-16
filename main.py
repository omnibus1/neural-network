import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)


training_input = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_output = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
weights = 2*np.random.random((3, 1))-1
print("weights:")
print(weights)
print("---------------")

for _ in range(20000):
    input_layer=training_input
    output=sigmoid(np.dot(input_layer,weights))
    error=training_output-output
    adjustments=error*sigmoid_derivative(output)
    weights+=np.dot(input_layer.T,adjustments)
print(output)
print("---------------")
print(weights)
