#train test split ->spits the data set into 2xx
from sklearn.model_selection import train_test_split
import numpy as np

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42) #test size=0.2=20%test and 80%train data,random state enusre unoiromity
print(f"training sets {X_train} and test sets {X_test}\n")
print(f"training sets {y_train} and test sets {y_test}")

'''
following outputs were obtianed
training sets [[8 9]
 [4 5]
 [0 1]
 [6 7]] and test sets [[2 3]]

training sets [4, 2, 0, 3] and test sets [1]
'''