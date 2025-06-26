from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from activations.activation_relu import Activation_Relu
from activations.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss

Wine = load_wine()
# print(X.shape,y.shape)

#reshaping y
X=Wine.feature_names
y=Wine.target.reshape(-1,1) #reshaping for one hot encoding
print(y.shape)

#one hot encoder
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)
print(y_encoded[1:3])


#spltting the data set in 80:20
X_train,X_test,y_train,y_test=train_test_split(X, y_encoded, test_size=0.2, random_state=25)


input_layer=Layer_Dense(13,10)
middle_layer=Layer_Dense(10,10)
output_layer=Layer_Dense(10,3)

relu=Activation_Relu()
softmax=Softmax_activation()

loss=CrossEntropyLoss()

lossess=[]
accuracies=[]

epoch=1000
lr=0.02

for epoch in range(epoch):
    #forward

    # input layer
    first_layer_input = input_layer.forward(X_train)
    first_relu_output = relu.forward(first_layer_input)  # relu out of input

    # middle layer
    second_layer_input = middle_layer.forward(first_relu_output)
    second_relu_output = relu.forward(second_layer_input)

    # output lauyer->softmax instead of relu
    final_layer_input = output_layer.forward(second_relu_output)
    final_softmax_output = softmax.forward(final_layer_input)


    #losses
