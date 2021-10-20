from io import BytesIO
import base64
import re

from PIL import Image
import numpy as np
import requests

from service.serving.registry import models_registry


def get_object_detection_prediction(model_name, image_b64=None, image_url=None,
                                    confidence_thresh=0.5, cls_model_name=None):
    model = models_registry[model_name]
    image = None
    if image_b64:
        image_data = re.sub('^data:image/.+;base64,', '', image_b64)
        image = Image.open(BytesIO(base64.b64decode(image_data)))
    elif image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    predictions = model.predict(image)
    if cls_model_name and len(predictions) > 0:
        cls_model = models_registry[cls_model_name]
        img_list = [np.array(image.crop(p["bbox"])) for p in predictions]
        cls_predictions = cls_model.predict(img_list)
        for i, pred in enumerate(predictions):
            pred["cls_score"] = cls_predictions[i]["cls_score"]
            pred["class_idx"] = cls_predictions[i]["class_idx"]
            pred["class_name"] = cls_predictions[i]["class_name"]
    return predictions
