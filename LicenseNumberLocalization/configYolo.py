import albumentations as A
import torch
import cv2

from albumentations.pytorch import ToTensorV2

# ---- Hyper Params ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
IMAGE_SIZE = 416
NUM_EPOCHS = 100
NUM_WORKERS = 12
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 3e-4
NMS_IOU_THRESHOLD = 0.4
NMS_PROB_THRESHOLD = 0.7
SCALES = (IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8)
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_FILE = r"ПУТЬ СОХРАНЕНИЯ ЧЕКПОИНТА"
DATASET_DIR     = r"ПУТЬ К КАТАЛОГУ С ИЗОБРАЖЕНИЯМИ"
LABEL_FILE      = r"ПУТЬ К JSON ФАЙЛУ С РАЗМЕТКОЙ"
ANCHORS = [
    [(0.38, 0.17), (0.50, 0.29), (0.9, 0.65)],
    [(0.16, 0.07), (0.15, 0.11), (0.35, 0.13)],
    [(0.03, 0.01), (0.07, 0.04), (0.08, 0.06)],
]

scale = 1.1

# Аугментация изображений при обучении
transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.5, hue=0.4, p=0.5),
        A.Blur(p=0.1),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.CLAHE(p=0.1),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
        ToTensorV2()
    ],
    bbox_params=A.BboxParams(format='yolo', min_visibility=0.4, label_fields=[])
)

# Аугментация изображений при работе
connect_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE),
            min_width=int(IMAGE_SIZE),
            border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255),
        ToTensorV2()
    ]
)
