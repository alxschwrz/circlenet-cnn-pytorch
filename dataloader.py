import torch
from torch.utils.data import Dataset
from skimage import io
import pandas as pd
import os

class SphereDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.labels = pd.read_csv(self.root_dir+csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.labels.iloc[idx, 0])
        image = io.imread(img_name)

        # We are only interested in the center coordinates
        center_coordinates = self.labels.iloc[idx, 1:3]
        center_coordinates = torch.from_numpy(center_coordinates.to_numpy().astype('float32'))

        if self.transform:
            image = self.transform(image)

        return image, center_coordinates
