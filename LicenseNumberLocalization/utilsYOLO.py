import torch


def intersection_over_union(bboxes1: torch.tensor, bboxes2: torch.tensor, mode='corners'):

    try:
        assert mode in ('midpoint', 'width-height', 'corners')
    except Exception:
        print('invalid intersection_over_union mode')

    # bbox format is [w, h]
    if mode == 'width-height':
        intersection = torch.min(bboxes1[..., 0], bboxes2[..., 0]) * torch.min(bboxes1[..., 1], bboxes2[..., 1])
        union = bboxes1[..., 0] * bboxes1[..., 1] + bboxes2[..., 0] * bboxes1[..., 1] - intersection + 1e-16
        iou = intersection/union
        return iou

    # bbox format is [x, y, w, h]
    if mode == 'midpoint':
        b1x1, b2x1 = bboxes1[..., 0] - bboxes1[..., 2]/2, bboxes2[..., 0] - bboxes2[..., 2]/2
        b1y1, b2y1 = bboxes1[..., 1] - bboxes1[..., 3]/2, bboxes2[..., 1] - bboxes2[..., 3]/2

        b1x2, b2x2 = bboxes1[..., 0] + bboxes1[..., 2]/2, bboxes2[..., 0] + bboxes2[..., 2]/2
        b1y2, b2y2 = bboxes1[..., 1] + bboxes1[..., 3]/2, bboxes2[..., 1] + bboxes2[..., 3]/2

        bboxes1[..., 0:4] = b1x1, b1y1, b1x2, b1y2
        bboxes2[..., 0:4] = b2x1, b2y1, b2x2, b2y2

        mode = 'corners'

    # bbox format is [x1, y1, x2, y2]
    # slice to keep the last dim
    if mode == 'corners':
        intersection_x1 = torch.max(bboxes1[..., 0:1], bboxes2[..., 0:1])
        intersection_y1 = torch.max(bboxes1[..., 1:2], bboxes2[..., 1:2])
        intersection_x2 = torch.min(bboxes1[..., 2:3], bboxes2[..., 2:3])
        intersection_y2 = torch.min(bboxes1[..., 3:4], bboxes2[..., 3:4])

    intersection = (intersection_x2 - intersection_x1).clamp(0) * (intersection_y2 - intersection_y1).clamp(0)
    bboxes1_square = bboxes1[..., 2] - bboxes1[..., 0] * bboxes1[..., 3] - bboxes1[..., 1]
    bboxes2_square = bboxes2[..., 2] - bboxes2[..., 0] * bboxes2[..., 3] - bboxes2[..., 1]
    union = bboxes1_square + bboxes2_square - intersection + 1e-16
    iou = intersection/union

    return iou
