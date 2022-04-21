import json
import logging
import os

from PIL import Image
import albumentations as A
import numpy as np
import torch
import torchvision

from ..tools.utils import rle_encoding


class ISModel:

    def __init__(self, model_path):
        self.model_path = model_path
        self.transforms = A.load(os.path.join(model_path, 'transforms.json'))
        with open(os.path.join(model_path, 'class_mapping.json')) as data:
            mappings = json.load(data)

        self.class_mapping = {item['model_idx']: item['class_name'] for item in mappings}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            logging.warning("GPU not found")
        model = torch.jit.load(os.path.join(model_path, 'model.pt'))
        self.model = model.to(self.device)
        logging.info(f"Model {model_path} loaded")

    def predict(self, image: Image, score_threshold: float = 0.5):
        width, height = image.width, image.height
        image = np.array(image)
        image = self.transforms(image=image)['image']
        x = torch.from_numpy(image)
        x = x.to(self.device)
        with torch.no_grad():
            # Convert to channels first, convert to float datatype
            x = x.permute(2, 0, 1).float()
            y = self.model(x)
            scale_factor = height / x.shape[1]
            to_keep = torchvision.ops.nms(y['pred_boxes'], y['scores'], 0.5)
            y['pred_boxes'] = y['pred_boxes'][to_keep]
            y['pred_classes'] = y['pred_classes'][to_keep]
            y['pred_masks'] = y['pred_masks'][to_keep]

        y['pred_classes'] = y['pred_classes'].cpu().numpy()
        y['pred_boxes'] = y['pred_boxes'].cpu().numpy()
        y['pred_masks'] = y['pred_masks'].cpu().numpy()
        y['scores'] = y['scores'].cpu().numpy()
        results = []
        for i in range(len(y['pred_classes'])):
            if y['scores'][i] < score_threshold:
                continue
            bbox = list(map(int, y['pred_boxes'][i] * scale_factor))
            class_idx = y['pred_classes'][i]
            mask = rle_encoding(y['pred_masks'][i])
            results.append({'bbox': bbox,
                            'mask': mask,
                            'score': y['scores'][i],
                            'class_idx': class_idx,
                            'class_name': self.class_mapping[class_idx]})
        return results
