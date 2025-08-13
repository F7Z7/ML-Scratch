import numpy as np
from sklearn.datasets import load_breast_cancer

from activations.activation_relu import Activation_Relu
from activations.activation_softmax import Softmax_activation
from layers.dense import Layer_Dense
from loss.cross_entropy import CrossEntropyLoss
from sklearn import datasets
from sklearn.model_selection import train_test_split  # for splitting the data to test and train
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

breast_cancer=load_breast_cancer()
X, y = load_breast_cancer(return_X_y=True)

print(f"Input feautres size {X.shape}")
print(f"Output feautres size {y.shape}")