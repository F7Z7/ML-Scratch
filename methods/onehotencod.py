from sklearn.preprocessing import OneHotEncoder
import numpy as np


y = np.array([0, 1, 2, 1, 0]).reshape(-1, 1)
print(y)
encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y)

print(y_encoded)
'''
output
[[0]
 [1]
 [2]
 [1]
 [0]]
 represented in onehot encoded form 
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
'''
