import torch
import torch.nn as nn
import string
from OCR import configOCR


class SimpleOCR(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.ModuleList(
            [
                # 256x32x144
                nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 256x16x72
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.Conv2d(64, 64, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 256x8x36
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, 128, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 256x4x18
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.Conv2d(256, 256, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(kernel_size=2, stride=2),
                # 256x2x9
            ]
        )
        self.out = nn.ModuleList(
            [
                nn.Linear((2 * 9 * 256), 64),
                nn.LeakyReLU(0.1),
                nn.Linear(64, 9 * (len(string.digits) + len(configOCR.LETTER_LIST)))
            ]
        )

    def forward(self, x):
        for module in self.conv:
            x = module(x)
        x = torch.flatten(x, start_dim=1)
        for module in self.out:
            x = module(x)
        return x.reshape(x.shape[0], 9, len(string.digits) + len(configOCR.LETTER_LIST))

