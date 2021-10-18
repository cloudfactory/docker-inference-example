import argparse
import logging
import os
import sys

import waitress
from flask import Flask

from service.endpoints.inference import inference_api
from service.serving.registry import models_registry

# Set up logger
log_format = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(stream=sys.stdout,
                    format=log_format,
                    level=logging.INFO)

# Flask app with database configuration.
app = Flask('Inference Service')
app.register_blueprint(inference_api)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument("--model_name", help="Model name")
    parser.add_argument("--cls_model_name", help="Classifiers model name")
    args = parser.parse_args()
    model_name = args.model_name
    cls_model_name = args.cls_model_name
    # TODO Only one OD is supported right now
    models_registry.load_model("detector", os.path.join('model', model_name), model_name)
    models_registry.load_model("classifier", os.path.join('model', cls_model_name), cls_model_name)
    waitress.serve(app, host='0.0.0.0', port='5000', threads=8)
