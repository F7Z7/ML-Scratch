from sklearn.datasets import load_wine, load_iris, load_breast_cancer


class BaseConfig:
    name = ""
    input_size = 0
    hidden_size = 100
    num_classes = 0
    epochs = 1000
    lr = 0.1
    scale = True

    @staticmethod
    def load_dataset():
        raise NotImplementedError


class WineConfig(BaseConfig):
    name = "Wine"
    input_size = 13
    num_classes = 3

    @staticmethod
    def load_dataset():
        data = load_wine()
        return data.data, data.target, data.target_names


class IrisConfig(BaseConfig):
    name = "Iris"
    input_size = 4
    hidden_size = 10
    num_classes = 3
    lr = 0.05
    scale = False

    @staticmethod
    def load_dataset():
        data = load_iris()
        return data.data, data.target, data.target_names


class BreastCancerConfig(BaseConfig):
    name = "Breast Cancer"
    input_size = 30
    num_classes = 2

    @staticmethod
    def load_dataset():
        data = load_breast_cancer()
        return data.data, data.target, data.target_names
