from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import imgaug
from imgaug.augmentables.bbs import BoundingBox

class CardiacDataset(Dataset):
    def __init__(self, path_to_labels_csv, patients, root_path, augs):
        super().__init__()
        self.labels = pd.read_csv(path_to_labels_csv)
        self.patients = np.load(patients)
        self.root_path = Path(root_path)
        self.augment = augs

    def __len__(self):
        return len(self.patients)
    
    def __getitem__(self, index):
        patient = self.patients[index]
        data = self.labels[self.labels['name'] == patient]

        x_min = data['x0'].item()
        y_min = data['y0'].item()
        x_max = x_min + data['w'].item()
        y_max = y_min + data['h'].item()
        bbox = [x_min, y_min, x_max, y_max]

        file_path = self.root_path / patient
        img = np.load(f'{file_path}.npy').astype(np.float32)

        if self.augment:
            bb = BoundingBox(x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3])
            random_seed = torch.randint(0, 100000, (1,)).item()
            imgaug.seed(random_seed)

            img, aug_bbox = self.augment(image=img, bounding_boxes = bb)
            bbox = aug_bbox[0][0], aug_bbox[0][1], aug_bbox[1][0], aug_bbox[1][1]

        img = (img - 0.49430165816326493) / 0.2527964897943661
        img = torch.tensor(img).unsqueeze(0)
        bbox = torch.tensor(bbox)

        return img, bbox
