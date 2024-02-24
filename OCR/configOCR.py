import albumentations as A
import torch
import cv2
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 6
BATCH_SIZE = 16
LETTER_LIST = ['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X', 'r', 'u', 's']
IMAGE_SIZE = 256
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
IMAGE_DIR = r'C:\Programming\Datasets\License Plate OCR'
MODEL_DIR = r'C:\Programming\Models\Vision\OCR_checkpoint.tar'

СONNECT_TRANSFORMS = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_width=IMAGE_SIZE, min_height=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(
            mean=0,
            std=1,
            always_apply=True
        ),
        ToTensorV2()
    ]
)

TRANSFORMS = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(min_width=IMAGE_SIZE, min_height=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5, hue=0.4, p=0.5),
        A.Blur(p=0.1),
        A.Affine(translate_percent=0.1, shear=(-45, 45), p=0.5),
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Posterize(p=0.1),
        A.CLAHE(p=0.1),
        A.Normalize(
            mean=0,
            std=1,
            always_apply=True
        ),
        ToTensorV2()
    ]
)