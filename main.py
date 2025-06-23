import numpy as np

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
# _,ax=plt.subplots()
# scatter = ax.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
# ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
# _ = ax.legend(
#     scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
# )
# # plt.show()
# encoder = OneHotEncoder(sparse_output=False)
# y_encoded = encoder.fit_transform(y)
# print(y_encoded)
#splitin datasets into 80:20 format
y=iris.target.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=25)

#3 layer network
input_layer=Layer_Dense(4,10)
middle_layer=Layer_Dense(10,10)
output_layer=Layer_Dense(10,3)

#activations
relu=Activation_Relu()
softmax=Softmax_activation()
#losses
loss=CrossEntropyLoss()

epoch=1000 #no of iterations
lr=0.5      #learning rate


for epoch in range(epoch):
    #input layer
    first_layer_input=input_layer.forward(X_train)
    first_relu_output=relu.forward(first_layer_input) #relu out of input

    #middle layer
    second_layer_input=middle_layer.forward(first_relu_output)
    second_relu_output=relu.forward(second_layer_input)

    #output lauyer->softmax instead of relu
    final_layer_input=output_layer.forward(second_relu_output)
    final_softmax_output=softmax.forward(final_layer_input)


    #losss
    output_loss=loss.forward(final_softmax_output, y_train)

    #accuracy
    prediction=np.argmax(final_softmax_output,axis=1)
    # true_classes=np.argmax(y_train,axis=1) #correct
    accuracy=np.mean(prediction== y_train.flatten())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {output_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")


    #backpropoagtion
    dloss=loss.backward(final_softmax_output,y_train)
    back_input=output_layer.backward(dloss,lr)

    drelu_middle=relu.backward(back_input)
    dmiddle=middle_layer.backward(drelu_middle,lr)

    drelu_input=relu.backward(dmiddle)
    dinput=input_layer.backward(drelu_input,lr)





