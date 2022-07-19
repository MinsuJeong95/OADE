import torch
import os
import PIL
from glob import glob


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.classes = os.listdir(self.root_dir)
        self.transforms = transforms
        self.data = []
        self.labels = []

        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)
            for img in glob(os.path.join(cls_dir, '*')):
                self.data.append(img)
                self.labels.append(idx)


    def __getitem__(self, idx):
        img_path, label = self.data[idx], self.labels[idx]
        img = PIL.Image.open(img_path)

        if self.transforms:
            img = self.transforms(img)

        sample = {'image': img, 'label': label, 'filename': img_path}

        return sample


    def __len__(self):
        return len(self.data)