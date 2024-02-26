import torch
import string
import configOCR
import os
import numpy as np
from PIL import Image
from PIL import ImageEnhance


class OCRDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, transfroms=None):
        self.image_dir = image_dir
        self.transforms = transfroms
        self.image_list = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(image_path).convert('L')
        image = np.array(image)

        label_string = self.image_list[idx].split('.')[0].split('_')[0]
        if len(label_string) < 9:
            label_string = label_string[0:6] + '0' + label_string[6:]
        label = torch.zeros(len(label_string), 1)

        if self.transforms:
            augmentations = self.transforms(image=image)
            image = augmentations['image']

        for i in range(len(label_string)):
            letter = label_string[i]
            if letter in configOCR.LETTER_LIST:
                label[i][0] = configOCR.LETTER_LIST.index(letter)
            elif letter in string.digits:
                label[i][0] = string.digits.index(letter) + len(configOCR.LETTER_LIST)
        return image, label, label_string
