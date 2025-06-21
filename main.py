from layers.activation_relu import Activation_Relu
from layers.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss
from sklearn import datasets


iris = datasets.load_iris()
print(iris.feature_names)  # 4inputs and 3 outpua
for elem in dir(iris):
    print(iris.elem)
input_layer=Layer_Dense(4,10)
middle_layer=Layer_Dense(10,10)
middle_layer2=Layer_Dense(10,10)
output_layer=Layer_Dense(10,3)