from io import BytesIO
import base64
import re

from PIL import Image
import requests

from service.serving.registry import models_registry


def get_object_detection_prediction(model_name, image_b64=None, image_url=None,
                                    confidence_thresh=0.5):
    model = models_registry[model_name]
    image = None
    if image_b64:
        image_data = re.sub('^data:image/.+;base64,', '', image_b64)
        image = Image.open(BytesIO(base64.b64decode(image_data)))
    elif image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
    predictions = model.predict(image)
    return predictions
