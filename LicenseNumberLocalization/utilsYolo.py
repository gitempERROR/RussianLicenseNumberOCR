import torch
from LicenseNumberLocalization import configYolo


def intersection_over_union(bboxes1: torch.tensor, bboxes2: torch.tensor, mode='corners'):
    try:
        assert mode in ('midpoint', 'width-height', 'corners')
    except Exception:
        print('invalid intersection_over_union mode')

    # bbox format is [w, h]
    if mode == 'width-height':
        intersection = torch.min(bboxes1[..., 0], bboxes2[..., 0]) * torch.min(bboxes1[..., 1], bboxes2[..., 1])
        union = bboxes1[..., 0] * bboxes1[..., 1] + bboxes2[..., 0] * bboxes1[..., 1] - intersection + 1e-16
        iou = intersection / union
        return iou

    # bbox format is [x, y, w, h]
    if mode == 'midpoint':
        bboxes1[..., 0], bboxes2[..., 0] = bboxes1[..., 0] - bboxes1[..., 2] / 2, bboxes2[..., 0] - bboxes2[..., 2] / 2
        bboxes1[..., 1], bboxes2[..., 1] = bboxes1[..., 1] - bboxes1[..., 3] / 2, bboxes2[..., 1] - bboxes2[..., 3] / 2

        bboxes1[..., 2], bboxes2[..., 2] = bboxes1[..., 0] + bboxes1[..., 2] / 2, bboxes2[..., 0] + bboxes2[..., 2] / 2
        bboxes1[..., 3], bboxes2[..., 3] = bboxes1[..., 1] + bboxes1[..., 3] / 2, bboxes2[..., 1] + bboxes2[..., 3] / 2

        mode = 'corners'

    # bbox format is [x1, y1, x2, y2]
    # slice to keep the last dim
    if mode == 'corners':
        intersection_x1 = torch.max(bboxes1[..., 0:1], bboxes2[..., 0:1])
        intersection_y1 = torch.max(bboxes1[..., 1:2], bboxes2[..., 1:2])
        intersection_x2 = torch.min(bboxes1[..., 2:3], bboxes2[..., 2:3])
        intersection_y2 = torch.min(bboxes1[..., 3:4], bboxes2[..., 3:4])

    intersection = (intersection_x2 - intersection_x1).clamp(0) * (intersection_y2 - intersection_y1).clamp(0)
    bboxes1_square = (bboxes1[..., 2:3] - bboxes1[..., 0:1]) * (bboxes1[..., 3:4] - bboxes1[..., 1:2])
    bboxes2_square = (bboxes2[..., 2:3] - bboxes2[..., 0:1]) * (bboxes2[..., 3:4] - bboxes2[..., 1:2])
    union = bboxes1_square + bboxes2_square - intersection + 1e-16
    iou = intersection / union

    return iou


def non_max_suppression(bboxes, iou_threshold, prob_threshold, mode="corners"):
    assert type(bboxes) is list

    bboxes_after_nms = []
    bboxes = [box for box in bboxes if box[0] >= prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[0], reverse=True)

    while bboxes:
        bboxes_after_nms.append(bboxes.pop(0))
        bboxes = [
            box for box in bboxes
            if
            (box[3] * box[4]) / (bboxes_after_nms[-1][4] * bboxes_after_nms[-1][4])
            > intersection_over_union(
                torch.tensor(box[1:]),
                torch.tensor(bboxes_after_nms[-1][1:]),
                mode=mode
            ) < iou_threshold
        ]

    return bboxes_after_nms


def get_loaders():
    from datasetYolo import YoloDataset

    train_dataset = YoloDataset(
        dataset_dir=configYolo.DATASET_DIR,
        label_file=configYolo.LABEL_FILE,
        anchors=configYolo.ANCHORS,
        transforms=configYolo.transforms
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        shuffle=True,
        num_workers=configYolo.NUM_WORKERS,
        batch_size=configYolo.BATCH_SIZE,
        drop_last=True
    )

    return train_dataloader


def anchor_scaler():
    scaled_anchors = (
            torch.tensor(configYolo.ANCHORS)
            * torch.tensor(configYolo.SCALES).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(configYolo.DEVICE)

    return scaled_anchors


def save_checkpoint(model, optimizer):
    print('Saving checkpoint...', end='')
    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, configYolo.CHECKPOINT_FILE)
    print('  --->  Save complete')


def load_checkpoint(model, optimizer):
    print('Loading checkpoint...', end='')
    checkpoint = torch.load(f=configYolo.CHECKPOINT_FILE, map_location=configYolo.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    for param_group in optimizer.param_groups:
        param_group['lr'] = configYolo.LEARNING_RATE
    print('  --->  Load complete')


def bboxes_conversion(bboxes_predictions: torch.tensor, split_size) -> list:
    split_size_id = configYolo.SCALES.index(split_size)
    anchors = anchor_scaler()[split_size_id]
    batch_size = bboxes_predictions.shape[0]
    num_anchors = len(anchors)
    anchors = anchors.reshape(1, num_anchors, 1, 1, 2)

    bboxes_predictions[..., 0:1] = torch.sigmoid(bboxes_predictions[..., 0:1])
    bboxes_predictions[..., 1:3] = torch.sigmoid(bboxes_predictions[..., 1:3])
    bboxes_predictions[..., 3:5] = torch.exp(bboxes_predictions[..., 3:5]) * anchors

    cell_indices = (
        torch.arange(split_size)
        .repeat(1, 3, split_size, 1)
        .unsqueeze(-1)
    ).to(bboxes_predictions.device)

    bboxes_predictions[..., 1:2] = 1 / split_size * (bboxes_predictions[..., 1:2] + cell_indices)
    cell_indices = cell_indices.permute(0, 1, 3, 2, 4)
    bboxes_predictions[..., 2:3] = 1 / split_size * (bboxes_predictions[..., 2:3] + cell_indices)
    bboxes_predictions[..., 3:5] = 1 / split_size * bboxes_predictions[..., 3:5]

    bboxes_predictions = bboxes_predictions.reshape(batch_size, num_anchors * (split_size ** 2), 5)

    return bboxes_predictions.tolist()
