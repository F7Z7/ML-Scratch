import numpy as np

class Softmax_activation:

    def softmax_forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True) #returns the value between 0 and 1
        self.outputs = probabilities
        return self.outputs
