from models.base_model import BaseNNModel
from models.configs import IrisConfig

def run():
    model = BaseNNModel(IrisConfig)
    model.load_data()
    model.train()
    model.evaluate()
