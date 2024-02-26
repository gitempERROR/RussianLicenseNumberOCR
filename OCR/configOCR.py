import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 12
BATCH_SIZE = 8
LETTER_LIST = ['A', 'B', 'E', 'K', 'M', 'H', 'O', 'P', 'C', 'T', 'Y', 'X']
IMAGE_SIZE = 144
IMAGE_HEIGHT = 32
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 3e-4
NUM_EPOCHS = 30
LOAD_MODEL = True
SAVE_MODEL = True
IMAGE_DIR = r'C:\Programming\Datasets\License Plate OCR'
MODEL_DIR = r'C:\Programming\Models\Vision\OCR_checkpoint.tar'

СONNECT_TRANSFORMS = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_SIZE),
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
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_SIZE, p=1),
        A.Affine(translate_percent=0.0, shear=(-45, 45), p=0.5),
        A.ColorJitter(contrast=0.2, always_apply=True),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Posterize(p=0.1),
        A.Normalize(
            mean=0,
            std=1,
            always_apply=True
        ),
        ToTensorV2()
    ]
)