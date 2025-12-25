import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from activations.activation_relu import Activation_Relu
from activations.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss


class BaseNNModel:
    def __init__(self, config):
        self.cfg = config
        self.relu = Activation_Relu()
        self.softmax = Softmax_activation()
        self.loss_fn = CrossEntropyLoss()

        self.losses = []
        self.accuracies = []

        self._build_model()

    def _build_model(self):
        self.input_layer = Layer_Dense(self.cfg.input_size, self.cfg.hidden_size)
        self.middle_layer = Layer_Dense(self.cfg.hidden_size, self.cfg.hidden_size)
        self.output_layer = Layer_Dense(self.cfg.hidden_size, self.cfg.num_classes)

    def load_data(self):
        X, y, target_names = self.cfg.load_dataset()

        if self.cfg.scale:
            X = StandardScaler().fit_transform(X)

        y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=25
        )

        self.target_names = target_names

    def forward(self, X):
        x = self.relu.forward(self.input_layer.forward(X))
        x = self.relu.forward(self.middle_layer.forward(x))
        x = self.output_layer.forward(x)
        return self.softmax.forward(x)

    def backward(self, y_pred):
        dloss = self.loss_fn.backward(y_pred, self.y_train)

        dx = self.output_layer.backward(dloss)
        dx = self.relu.backward(dx)

        dx = self.middle_layer.backward(dx)
        dx = self.relu.backward(dx)

        self.input_layer.backward(dx)

        for layer in [self.output_layer, self.middle_layer, self.input_layer]:
            layer.update_params(self.cfg.lr)

    def train(self):
        for epoch in range(self.cfg.epochs):
            y_pred = self.forward(self.X_train)

            loss = self.loss_fn.forward(y_pred, self.y_train)
            acc = np.mean(
                np.argmax(y_pred, axis=1) == np.argmax(self.y_train, axis=1)
            )

            self.losses.append(loss)
            self.accuracies.append(acc)

            if epoch % 100 == 0:
                print(f"[{self.cfg.name}] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc*100:.2f}%")

            self.backward(y_pred)

    def evaluate(self):
        y_pred = self.forward(self.X_test)
        preds = np.argmax(y_pred, axis=1)
        true = np.argmax(self.y_test, axis=1)

        print(f"\n {self.cfg.name} Test Accuracy: {np.mean(preds == true)*100:.2f}%")
        print(classification_report(true, preds, target_names=self.target_names))

        cm = confusion_matrix(true, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=self.target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(self.cfg.name)
        plt.show()
