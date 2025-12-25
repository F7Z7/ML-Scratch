from models.base_model import BaseNNModel
from models.configs import WineConfig

def run():
    model = BaseNNModel(WineConfig)
    model.load_data()
    model.train()
    model.evaluate()
