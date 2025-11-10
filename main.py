import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib
matplotlib.use('TkAgg')
from activations.activation_relu import Activation_Relu
from activations.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss
from sklearn.model_selection import train_test_split  # for splitting the data to test and train
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay


import matplotlib.pyplot as plt

breast_can = load_breast_cancer()

X = breast_can.data
y = breast_can.target.reshape(-1, 1)  # reshaping for one hot encoding
print(y.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) #scales the input

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=25)

# 3 layer network
input_layer = Layer_Dense(30, 100)
middle_layer = Layer_Dense(100, 100)
output_layer = Layer_Dense(100, 2)

# activations
relu = Activation_Relu()
softmax = Softmax_activation()
# losses
loss = CrossEntropyLoss()
losses = []
accuracies = []
epoch = 1000  # no of iterations
lr = .1  # learning rate

for epoch in range(epoch):
    # input layer
    first_layer_input = input_layer.forward(X_train)
    first_relu_output = relu.forward(first_layer_input)  # relu out of input

    # middle layer
    second_layer_input = middle_layer.forward(first_relu_output)
    second_relu_output = relu.forward(second_layer_input)

    # output lauyer->softmax instead of relu
    final_layer_input = output_layer.forward(second_relu_output)
    final_softmax_output = softmax.forward(final_layer_input)

    # losss
    output_loss = loss.forward(final_softmax_output, y_train)

    # accuracy
    prediction = np.argmax(final_softmax_output, axis=1)
    # true_classes=np.argmax(y_train,axis=1) #correct
    accuracy = np.mean(np.argmax(final_softmax_output, axis=1) == np.argmax(y_train, axis=1))

    losses.append(output_loss)
    accuracies.append(accuracy)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {output_loss:.4f}, Accuracy: {accuracy * 100:.2f}%")

    # backpropoagtion
    dloss = loss.backward(final_softmax_output, y_train)
    back_input = output_layer.backward(dloss)

    drelu_middle = relu.backward(back_input)
    dmiddle = middle_layer.backward(drelu_middle)

    drelu_input = relu.backward(dmiddle)
    dinput = input_layer.backward(drelu_input)

    for layers in [output_layer, middle_layer, input_layer]:
        layers.update_params(lr)

final_avg_accuray = accuracy.mean()
print(f"Final average accuracy is {final_avg_accuray * 100:.2f}")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))

# Plot Loss
axs[0,0].plot(losses, label='Loss', color='red')
axs[0,0].set_title("Loss over Epochs")
axs[0,0].set_xlabel("Epoch")
axs[0,0].set_ylabel("Loss")
axs[0,0].grid(True)

# Plot Accuracy
axs[0,1].plot([acc * 100 for acc in accuracies], label='Accuracy', color='green')  # Multiply by 100 for %
axs[0,1].set_title("Accuracy over Epochs")
axs[0,1].set_xlabel("Epoch")
axs[0,1].set_ylabel("Accuracy (%)")
axs[0,1].grid(True)


# testing of datasets

test_out1 = input_layer.forward(X_test)
test_relu1 = relu.forward(test_out1)
test_out2 = middle_layer.forward(test_relu1)
test_relu2 = relu.forward(test_out2)
test_out3 = output_layer.forward(test_relu2)
test_final_output = softmax.forward(test_out3)

test_prediction = np.argmax(test_final_output, axis=1)
test_true_labels = y_test.argmax(axis=1)
test_accuracy = np.mean(test_prediction == test_true_labels)

print(f"\n Final Test Accuracy: {test_accuracy * 100:.2f}%")

# confusion matrices
Confusion_Matrix = confusion_matrix(test_true_labels, test_prediction)
print(Confusion_Matrix)
disp = ConfusionMatrixDisplay(confusion_matrix=Confusion_Matrix,display_labels=breast_can.target_names)
disp.plot(ax=axs[1,0],cmap=plt.cm.Blues,colorbar=False)
axs[1,0].set_title("Confusion Matrix")
plt.subplots_adjust(wspace=0.4, hspace=0.6)
plt.show()
print(
    f"Classification Report{classification_report(test_true_labels, test_prediction, target_names=breast_can.target_names)}")
