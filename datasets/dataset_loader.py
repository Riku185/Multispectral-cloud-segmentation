
import torch
from torch.utils.data import Dataset
import tifffile as tiff
import numpy as np
import random

SELECTED_BANDS = [1,2,3,7]

class CloudDataset(Dataset):

    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def augment_pair(self, img, mask):

        if random.random() > 0.5:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=0)

        if random.random() > 0.5:
            img = np.flip(img, axis=2)
            mask = np.flip(mask, axis=1)

        return img.copy(), mask.copy()

    def __getitem__(self, idx):

        img = tiff.imread(self.image_paths[idx])
        img = img[SELECTED_BANDS,:,:]
        img = img.astype(np.float32) / 10000.0

        mask = tiff.imread(self.mask_paths[idx])
        mask = tiff.imread(self.mask_paths[idx])

        new_mask = np.zeros_like(mask)
        # clear
        new_mask[mask == 128] = 0
        # cloud shadow
        new_mask[mask == 64] = 1
        # cloud
        new_mask[mask == 255] = 2
        mask = new_mask.astype(np.int64)

        if self.augment:
            img, mask = self.augment_pair(img, mask)

        img = torch.tensor(img).float()
        mask = torch.tensor(mask).long()

        return img, mask
