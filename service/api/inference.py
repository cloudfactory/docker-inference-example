from io import BytesIO
import base64
import re
import logging

from PIL import Image
import numpy as np
import requests

from service.serving.registry import models_registry


def get_object_detection_prediction(model_name, image_b64=None, image_url=None,
                                    confidence_thresh=0.5, cls_model_name=None, attr_model_name=None):
    logging.debug(f"Trying to retrieve  object detection predictions for model {model_name}")
    model = models_registry[model_name]
    image = None
    if image_b64:
        logging.debug("Using image provided in base64 format")
        image_data = re.sub('^data:image/.+;base64,', '', image_b64)
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        logging.debug(f"Image decoded successfully {image.mode} {image.size}")
    elif image_url:
        logging.debug("Extracting image by URL")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        logging.debug(f"Image extracted successfully {image.mode} {image.size}")
    if image.mode != 'RGB':
        image = image.convert('RGB')
    logging.debug("Performing inference")
    predictions = model.predict(image)
    logging.debug(f"Predictions: {predictions}")
    if cls_model_name and len(predictions) > 0:
        logging.debug(f"Use classifier model to polish classes ({cls_model_name})")
        cls_model = models_registry[cls_model_name]
        img_list = [np.array(image.crop(p["bbox"])) for p in predictions]
        cls_predictions = cls_model.predict(img_list)
        logging.debug(f"CLS predictions ({cls_predictions})")
        for i, pred in enumerate(predictions):
            pred["cls_score"] = cls_predictions[i]["cls_score"]
            pred["class_idx"] = cls_predictions[i]["class_idx"]
            pred["class_name"] = cls_predictions[i]["class_name"]
    if attr_model_name and len(predictions) > 0:
        logging.debug(f"Use attributer model to extract object attribtues ({attr_model_name})")
        attr_model = models_registry[attr_model_name]
        img_list = [np.array(image.crop(p["bbox"])) for p in predictions]
        attr_predictions = attr_model.predict(img_list)
        logging.debug(f"Attribute predictions ({attr_predictions})")
        for i, pred in enumerate(predictions):
            pred["attributes"] = attr_predictions[i]["attributes"]
    logging.debug(f"Response: {predictions}")
    return predictions
