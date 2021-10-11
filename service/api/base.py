import json

from flask import Response
import numpy as np


class JSONEncoder(json.JSONEncoder):
    """
    Wrapper class to try calling an object's tojson() method.
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, list):
            return obj
        try:
            return obj.tojson()
        except AttributeError:
            return json.JSONEncoder.default(self, obj)


def get_json_response(data, status=200):
    response = Response(json.dumps(data, cls=JSONEncoder), status=status)
    response.mimetype = 'application/json'
    return response
