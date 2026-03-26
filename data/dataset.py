import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_image


class MRIDataset(Dataset):

    def __init__(self, root_dir, split="train", img_size=224, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.img_size = img_size
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ])

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)

            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)

                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)

        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        image = preprocess_image(image)
        image = image.astype(np.float32)

        image = np.transpose(image, (2, 0, 1))

        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        return self.classes