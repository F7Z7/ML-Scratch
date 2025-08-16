# from sklearn.datasets import make_moons
from nnfs.datasets import spiral_data
from sklearn.datasets import load_iris,load_wine,load_breast_cancer


# def load_data():
#     # X,y=make_moons(n_samples=100,random_state=10 )
#     data = load_breast_cancer()
#     # X, y = load_breast_cancer( return_X_y=False, as_frame=False)
#     print(data.feature_names)
data = load_breast_cancer()
print(data.target_names)