from layers.activation_relu import Activation_Relu
from layers.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss
from sklearn import datasets
from sklearn.model_selection import train_test_split #for splitting the data to test and train
from sklearn.preprocessing import OneHotEncoder


import matplotlib.pyplot as plt

iris = datasets.load_iris()
x=iris.data
y=iris.target.reshape(-1,1)
# print(x,y)
# print(iris.feature_names)  # 4inputs and 3 outpua

#plot scattering
_,ax=plt.subplots()
scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
_ = ax.legend(
    scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
)
plt.show()
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)
# print(y_encoded)
#splitin datasets into 80:20 format
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

#3 layer network
input_layer=Layer_Dense(4,10)
middle_layer=Layer_Dense(10,10)
output_layer=Layer_Dense(10,3)

