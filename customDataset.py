import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.io

class projectDataset(Dataset):
    """Custom Dataset for the project.

    Args:
        csv_file (str): Path to the CSV file with annotations.
        img_dir (str): Path to the directory containing the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        target_transform (callable, optional): Optional transform to be applied on a label.
    """
    def __init__(self, csv_file, img_dir, transform = None, target_transform = None):
        
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """Returns the sample at index 'idx'.

        Args:
            idx (int): The index of the sample to return.

        Returns:
            tuple: A tuple containing the index, image and label of the sample at index 'idx'.
        """
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = torchvision.io.read_image(img_path,torchvision.io.ImageReadMode.RGB)
        label = self.annotations.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return idx, image, label
