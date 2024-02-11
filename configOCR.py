import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2

DEVICE = 'Cuda' if torch.cuda.is_available() else 'Cpu'
NUM_WORKERS = 6
BATCH_SIZE = 6
LEARNING_RATE = 3e-4
NUM_EPOCHS = 100
IMAGE_DIR = r'C:\Programming\Datasets\Test fixed OCR dataset'
MODEL_DIR = r'C:\Programming\Models\Vision\OCR_checkpoint.tar'

TRANSFORMS = A.Compose(
    [
        A.Normalize(
            mean=(0, 0, 0),
            std=(1, 1, 1),
            always_apply=True
        ),
        ToTensorV2()
    ]
)
