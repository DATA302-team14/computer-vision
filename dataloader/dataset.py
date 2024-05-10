import cv2
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

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