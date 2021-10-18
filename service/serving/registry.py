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


models_registry = ModelRegistry()
