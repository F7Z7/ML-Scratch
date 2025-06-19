# from sklearn.datasets import make_moons
from nnfs.datasets import spiral_data


def load_data():
    # X,y=make_moons(n_samples=100,random_state=10 )
    X, y = spiral_data(samples=100,noise=0.1)
    return X,y
