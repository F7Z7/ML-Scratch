from models.base_model import BaseNNModel
from models.configs import MNISTConfig


def run():
    model= BaseNNModel(MNISTConfig)
    model.load_data()
    model.train()
    model.evaluate()
    # model.show_results()