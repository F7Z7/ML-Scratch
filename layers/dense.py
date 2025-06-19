import numpy as np
class Layer_Dense:
    def __init__(self,inputs,neurons):
        self.weights=np.random.randn(inputs,neurons)*np.sqrt(2/inputs)
        self.biases=np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs=inputs
        self.output=np.dot(self.inputs,self.weights)+self.biases #y=mx+c

    def backward(self,dvalues,lr):
        self.dweights=np.dot(self.inputs.T, dvalues)
        self.dbiases=np.sum(dvalues,axis=0) #summing all biases

        #new weight=pld weight-lr*loss
        self.weights-=lr*self.dweights
        self.biases-=lr*self.dbiases

