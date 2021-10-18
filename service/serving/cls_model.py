from typing import List, Union
import json
import logging
import os

from torch.utils.data import DataLoader, TensorDataset
import albumentations as A
import numpy as np
import torch
import torchvision


class CLSModel:

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

    def predict(self, images: List[np.array], batch_size=32):
        images_tensor = []
        for i in images:
            tr_img = self.transforms(image=i)['image']
            tr_img = torch.from_numpy(tr_img).permute(2, 0, 1)
            images_tensor.append(tr_img.unsqueeze(dim=0))
        images_tensor = torch.cat(images_tensor).float()
        images_dataset = TensorDataset(images_tensor)
        images_loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            preds = []
            for (batch,) in images_loader:
                batch = batch.to(self.device)
                preds = preds + list(torch.softmax(self.model(batch), dim=1).cpu().numpy())
        scores = np.max(np.array(preds), axis=1)
        indices = np.argmax(preds, 1)
        results = []
        for idx, score in zip(indices, scores):
            results.append({"cls_score": score,
                            "class_idx": idx,
                            "class_name": self.class_mapping[idx]})
        return results
