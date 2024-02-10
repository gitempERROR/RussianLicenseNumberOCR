import torch
import string
import os
import numpy as np
from PIL import Image


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transfroms=None):
        self.image_dir = image_dir
        self.transforms = transfroms
        self.image_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        image = np.array(Image.open(image_path).convert('RGB'))

        label_string = self.image_list[idx].split('_')[1].split('.')[0]
        label = torch.zeros(len(label_string), 1)

        if self.transforms:
            augmentations = self.transforms(image=image)
            image = augmentations['image']

        for i in range(len(label_string)):
            letter = label_string[i]
            if letter in string.ascii_letters:
                label[i][0] = string.ascii_letters.index(letter)
            elif letter in string.digits:
                label[i][0] = string.digits.index(letter)
        return image, label
