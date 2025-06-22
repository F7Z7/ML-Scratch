from layers.activation_relu import Activation_Relu
from layers.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss
from sklearn import datasets
from sklearn.model_selection import train_test_split #for splitting the data to test and train


import matplotlib.pyplot as plt

iris = datasets.load_iris()
print(iris.feature_names)  # 4inputs and 3 outpua

#plot scattering
_,ax=plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()
input_layer=Layer_Dense(4,10)
middle_layer=Layer_Dense(10,10)
middle_layer2=Layer_Dense(10,10)
output_layer=Layer_Dense(10,3)