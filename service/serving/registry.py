import logging
import os

from omegaconf import OmegaConf

from .attr_model import ATTRModel
from .cls_model import CLSModel
from .od_model import ODModel
from .ses_model import SESModel
from .tag_model import TAGModel


class ModelRegistry:

    def __init__(self):
        self.loaded_models = {}

    def load_model(self, model_family, model_path, model_name):
        if model_family == "detector":
            self.loaded_models[model_name] = ODModel(model_path)
        elif model_family == "classifier":
            self.loaded_models[model_name] = CLSModel(model_path)
        elif model_family == "attributer":
            self.loaded_models[model_name] = ATTRModel(model_path)
        elif model_family == "image-tagger":
            self.loaded_models[model_name] = TAGModel(model_path)
        elif model_family == "semantic-segmentor":
            self.loaded_models[model_name] = SESModel(model_path)
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
