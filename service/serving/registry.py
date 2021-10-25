import logging
import os

from omegaconf import OmegaConf

from .cls_model import CLSModel
from .od_model import ODModel


class ModelRegistry:

    def __init__(self):
        self.loaded_models = {}

    def load_model(self, model_family, model_path, model_name):
        if model_family == "detector":
            self.loaded_models[model_name] = ODModel(model_path)
        elif model_family == "classifier":
            self.loaded_models[model_name] = CLSModel(model_path)
        else:
            raise ValueError(f"Unsupported model family - {model_family}")

    def __getitem__(self, item):
        return self.loaded_models[item]

    @staticmethod
    def get_model_family(model_path):
        config_path = os.path.join(model_path, 'config.yaml')
        if os.path.isfile(config_path):
            conf = OmegaConf.load(config_path)
            global_key = conf.GLOBAL
            if global_key is None:
                logging.warning("config.yaml should contain key GLOBAL")
                return None
            return global_key.FAMILY
        else:
            logging.warning(f"config.yaml not found in folder {model_path}")
        return None


models_registry = ModelRegistry()
