import json
import logging
import os

from PIL import Image
import albumentations as A
import numpy as np
import torch
import torchvision

from ..tools.utils import mask2bbox, rle_encoding


class SESModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.transforms = A.load(os.path.join(model_path, "transforms.json"))
        with open(os.path.join(model_path, "class_mapping.json")) as data:
            mappings = json.load(data)

        self.class_mapping = {
            item["model_idx"]: item["class_name"] for item in mappings
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            logging.warning("GPU not found")
        model = torch.jit.load(os.path.join(model_path, "model.pt"))
        self.model = model.to(self.device)
        logging.info(f"Model {model_path} loaded")

    def predict(self, image: Image):
        width, height = image.width, image.height
        image = np.array(image)
        image = self.transforms(image=image)["image"]
        x = torch.from_numpy(image)
        x = x.to(self.device)
        with torch.no_grad():
            # Convert to channels first, convert to float datatype
            x = x.permute(2, 0, 1).unsqueeze(dim=0).float()
            y = self.model(x)
            y = torch.nn.functional.interpolate(
                y, size=(height, width), mode="bilinear", align_corners=False
            )[0]
            mask = np.argmax(y.cpu().numpy(), axis=0).astype(np.uint8)

        post_processed_class_preds = []
        indices = []
        bboxes = []
        for class_ind in range(y.shape[0]):
            pred = (mask == class_ind + 1).astype(np.uint8)
            if 1 not in pred:
                continue
            indices.append(class_ind)
            bbox = mask2bbox(pred)
            cropped_pred = pred[bbox[0]: bbox[1], bbox[2]: bbox[3]].copy()
            cropped_pred = rle_encoding(cropped_pred)
            post_processed_class_preds.append(cropped_pred)
            bboxes.append([bbox[2], bbox[0], bbox[3], bbox[1]])

        results = {
            "indices": indices,
            "boxes": bboxes,
            "rle_masks": post_processed_class_preds,
        }
        return results
