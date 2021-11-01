from flask import Blueprint, request

from .. import api

inference_api = Blueprint('inference_api', __name__)


@inference_api.route('/v1/object_detection', methods=['POST'])
def get_object_detection_prediction():
    confidence_thresh = request.json.get('confidence_threshold', 0.5)
    image = request.json.get('image', {})
    image_b64, image_url = None, None
    if 'b64' in image:
        image_b64 = image.get("b64")
    if 'url' in image:
        image_url = image.get("url")
    if not image_b64 and not image_url:
        raise ValueError("Image url or base64 should be provided")
    model = request.json.get('model', None)
    cls_model_name = request.json.get('cls_model_name', None)
    attr_model_name = request.json.get('attr_model_name', None)
    results = api.inference.get_object_detection_prediction(model, image_b64, image_url, confidence_thresh,
                                                            cls_model_name, attr_model_name)
    return api.base.get_json_response(results)
