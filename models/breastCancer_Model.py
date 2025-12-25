from models.base_model import BaseNNModel
from models.configs import BreastCancerConfig

def run():
    model = BaseNNModel(BreastCancerConfig)
    model.load_data()
    model.train()
    model.evaluate()
