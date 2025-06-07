from sklearn.datasets import make_moons


def load_data():
    X,y=make_moons(n_samples=100,random_state=10 )
    return X,y
