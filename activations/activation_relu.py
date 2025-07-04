import numpy as np

class Activation_Relu:
    def forward(self,inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0,inputs) #relu
        return self.outputs
    def backward(self,dvalues):
        dinputs = dvalues.copy()
        dinputs[self.inputs <= 0] = 0 #if less than 0=0,else the value passes
        return dinputs