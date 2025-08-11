import numpy as np


class Layer_Dense:
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons) * np.sqrt(2 / inputs)
        self.biases = np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(self.inputs, self.weights) + self.biases  # y=mx+c

        return self.output

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)

        # new weight=pld weight-lr*loss


        return self.dinputs

    def update_params(self, lr):
        self.weights -= lr * self.dweights
        self.biases -= lr * self.dbiases

