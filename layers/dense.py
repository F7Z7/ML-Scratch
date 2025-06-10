import numpy as np
class Layer_Dense:
    def __init__(self,inputs,neurons):
        self.weights=np.random.randn(inputs,neurons)*np.sqrt(2/inputs)
        self.biases=np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs=inputs
        self.output=np.dot(self.inputs,self.weights)+self.biases #y=mx+c



