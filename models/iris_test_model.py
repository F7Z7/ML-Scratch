import numpy as np

from layers.activation_relu import Activation_Relu
from layers.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss
from sklearn import datasets
from sklearn.model_selection import train_test_split #for splitting the data to test and train
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix


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
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)
# print(y_encoded)
#splitin datasets into 80:20 format
# y=iris.target.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=25)

#3 layer network
input_layer=Layer_Dense(4,10)
middle_layer=Layer_Dense(10,10)
output_layer=Layer_Dense(10,3)

#activations
relu=Activation_Relu()
softmax=Softmax_activation()
#losses
loss=CrossEntropyLoss()
losses=[]
accuracies=[]
epoch=1000 #no of iterations
lr=0.05      #learning rate


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
    accuracy=np.mean(np.argmax(final_softmax_output, axis=1) == np.argmax(y_train, axis=1))

    losses.append(output_loss)
    accuracies.append(accuracy)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {output_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")



    #backpropoagtion
    dloss=loss.backward(final_softmax_output,y_train)
    back_input=output_layer.backward(dloss,lr)

    drelu_middle=relu.backward(back_input)
    dmiddle=middle_layer.backward(drelu_middle,lr)

    drelu_input=relu.backward(dmiddle)
    dinput=input_layer.backward(drelu_input,lr)

final_avg_accuray=accuracy.mean()
print(f"Final average accuracy is {final_avg_accuray * 100:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Plot Loss

axs[0].plot(losses, label='Loss', color='red')
axs[0].set_title("Loss over Epochs")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].grid(True)

# Plot Accuracy
axs[1].plot([acc * 100 for acc in accuracies], label='Accuracy', color='green')  # Multiply by 100 for %
axs[1].set_title("Accuracy over Epochs")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy (%)")
axs[1].grid(True)

plt.tight_layout()
plt.show()
#testing of datasets

test_out1=input_layer.forward(X_test)
test_relu1=relu.forward(test_out1)
test_out2=middle_layer.forward(test_relu1)
test_relu2=relu.forward(test_out2)
test_out3=output_layer.forward(test_relu2)
test_final_output=softmax.forward(test_out3)


test_prediction=np.argmax(test_final_output,axis=1)
test_true_labels=y_test.argmax(axis=1)
test_accuracy=np.mean(test_prediction==test_true_labels)

print(f"\n✅ Final Test Accuracy: {test_accuracy * 100:.2f}%")

#confusion matrices
Confusion_Matrix=confusion_matrix(test_true_labels, test_prediction)
print(Confusion_Matrix)

print(f"Classification Report{classification_report(test_true_labels, test_prediction, target_names=iris.target_names)}")
