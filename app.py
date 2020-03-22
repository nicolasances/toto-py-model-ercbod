from flask import Flask
from model.ercbod import ERCBOD

from totoml.controller import ModelController
from totoml.config import ControllerConfig

app = Flask(__name__)

model_controller = ModelController(ERCBOD(), app, ControllerConfig(enable_batch_predictions_events=False, enable_single_prediction_events=False))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)