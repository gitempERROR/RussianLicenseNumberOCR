import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 1
BATCH_SIZE = 16
IMAGE_SIZE = 500
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
LOAD_MODEL = True
SAVE_MODEL = True
IMAGE_DIR = r'C:\Programming\Datasets\Test fixed OCR dataset'
MODEL_DIR = r'C:\Programming\Models\Vision\OCR_checkpoint.tar'

TRANSFORMS = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.Normalize(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            always_apply=True
        ),
        ToTensorV2()
    ]
)
