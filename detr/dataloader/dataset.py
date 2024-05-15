import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

import os
import json
import util.misc as utils

try:
    from panopticapi.evaluation import pq_compute
except ImportError:
    pass

class CustomDataset(Dataset):
    def __init__(self, data_root, csv_file, transform=None, infer=False):
        self.data_root = data_root
        self.data = pd.read_csv(os.path.join(data_root, csv_file))
        self.transform = transform
        self.infer = infer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1][2:]
        image = cv2.imread(os.path.join(self.data_root, img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image
        
        mask_path = self.data.iloc[idx, 2][2:]
        mask = cv2.imread(os.path.join(self.data_root, mask_path), cv2.IMREAD_GRAYSCALE)
        mask[mask == 255] = 12 #배경을 픽셀값 12로 간주

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

class PanopticEvaluator(object):
    def __init__(self, ann_file, ann_folder, output_dir="panoptic_eval"):
        self.gt_json = ann_file
        self.gt_folder = ann_folder
        if utils.is_main_process():
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
        self.output_dir = output_dir
        self.predictions = []

    def update(self, predictions):
        for p in predictions:
            with open(os.path.join(self.output_dir, p["file_name"]), "wb") as f:
                f.write(p.pop("png_string"))

        self.predictions += predictions

    def synchronize_between_processes(self):
        all_predictions = utils.all_gather(self.predictions)
        merged_predictions = []
        for p in all_predictions:
            merged_predictions += p
        self.predictions = merged_predictions

    def summarize(self):
        if utils.is_main_process():
            json_data = {"annotations": self.predictions}
            predictions_json = os.path.join(self.output_dir, "predictions.json")
            with open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))
            return pq_compute(self.gt_json, predictions_json, gt_folder=self.gt_folder, pred_folder=self.output_dir)
        return None