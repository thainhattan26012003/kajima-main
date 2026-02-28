from collections import Counter
from PIL import Image
import os
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np


class KajimaDataset(Dataset):
    def __init__(self, data_dir: str, transforms=None, target_transforms=None):
        self.transforms = transforms
        self.target_transforms = target_transforms

        self.image_labels = []
        for date in os.listdir(data_dir):
            date_folder_path = Path(data_dir) / date

            for img_dn in os.listdir(date_folder_path):
                img_dir = date_folder_path / img_dn
                try:
                    if os.path.isdir(img_dir):
                        _, label, _ = img_dn.split("-")
                        label = int(label) - 1
                        for img_fp in os.listdir(img_dir):
                            if os.path.isfile(img_dir / img_fp):
                                self.image_labels.append((img_dir / img_fp, label))
                except Exception as ex:
                    print("Failed folder: ", img_dir, ex)

        self.num_classes = len(set([label for _, label in self.image_labels]))

        counter = Counter([label for _, label in self.image_labels])
        sorted_idx = sorted(list(counter.keys()))
        self.cls_num_list = [counter[idx] for idx in sorted_idx]

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_fp, label = self.image_labels[idx]

        image = Image.open(image_fp)
        image = image.convert("RGB")
        image = np.array(image)
        # image = rescale(image, RESCALE_FACTOR)
        if self.transforms:
            image = self.transforms(image)
        if self.target_transforms:
            label = self.target_transforms(label)

        return image, label
