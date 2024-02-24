import torch
import numpy as np
import os
import json
# import xml.etree.ElementTree as ET

from PIL import Image
from utilsYolo import intersection_over_union

# class YoloDataset(torch.utils.data.Dataset):
#     def __init__(self, image_dir, label_dir, anchors, scales=(13, 26, 52), transforms=None):
#         self.image_dir = image_dir
#         self.image_list = os.listdir(image_dir)
#         self.label_dir = label_dir
#         self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
#         self.anchors_per_scale = self.anchors.shape[0] // len(scales)
#         self.scales = scales
#         self.ignore_iou_threshold = 0.5
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.image_list)
#
#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.image_list[idx])
#         label_path = os.path.join(self.label_dir, self.image_list[idx][0:-4] + '.xml')
#
#         image = np.array(Image.open(image_path).convert('RGB'))
#         tree = ET.parse(label_path)
#         root = tree.getroot()
#         bboxes = []
#
#         x1 = int(root[4][5][0].text)
#         y1 = int(root[4][5][1].text)
#         x2 = int(root[4][5][2].text)
#         y2 = int(root[4][5][3].text)
#
#         x1, x2 = x1 / image.shape[1], x2 / image.shape[1]
#         y1, y2 = y1 / image.shape[0], y2 / image.shape[0]
#
#         x = (x1 + x2) / 2
#         y = (y1 + y2) / 2
#         w = x2 - x1
#         h = y2 - y1
#
#         box = [x, y, w, h]
#         bboxes.append(box)
#
#         if self.transforms:
#             augmentations = self.transforms(image=image, bboxes=bboxes)
#             image = augmentations['image']
#             bboxes = augmentations['bboxes']
#
#         targets = [torch.zeros(self.anchors_per_scale, scale, scale, 5) for scale in self.scales]
#
#         for box in bboxes:
#             iou = intersection_over_union(torch.tensor(box[2:4]), self.anchors, mode="width-height")
#             anchor_indices = iou.argsort(descending=True, dim=0)
#             x, y, w, h = box
#
#             has_anchor = [False, False, False]
#
#             for anchor_id in anchor_indices:
#                 scale_id = anchor_id // self.anchors_per_scale
#                 on_scale_id = anchor_id % self.anchors_per_scale
#
#                 scale = self.scales[scale_id]
#                 i, j = int(y * scale), int(x * scale)
#
#                 anchor_taken = targets[scale_id][on_scale_id, i, j, 0]
#
#                 if anchor_taken:
#                     continue
#                 elif has_anchor[scale_id]:
#                     if iou[anchor_id] > self.ignore_iou_threshold:
#                         targets[scale_id][on_scale_id, i, j, 0] = -1
#                     continue
#
#                 has_anchor[scale_id] = True
#                 x_cell, y_cell = x * scale - j, y * scale - i
#                 w_cell, h_cell = w * scale, h * scale
#
#                 target = torch.tensor((1, x_cell, y_cell, w_cell, h_cell))
#                 targets[scale_id][on_scale_id, i, j, 0:5] = target
#
#         return image, tuple(targets)


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, label_file, anchors, transforms=None, scales=(13, 26, 52)):
        self.scales = scales
        self.transforms = transforms
        self.label_file = label_file
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.anchors_per_scale = self.anchors.shape[0] // len(self.scales)
        with open(self.label_file, 'r') as label:
            self.json = json.load(label)
        self.image_count = len(self.json['items'])
        self.image_list = [str(os.path.join(dataset_dir, json_item['file'])) for json_item in self.json['items']]
        self.ignore_threshold = 0.5

    def __len__(self):
        return self.image_count

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_list[idx]).convert('RGB'))
        bboxes = []
        for bbox in self.json['items'][idx]['nums']:
            x1 = abs(min((bbox['box'][0][0], bbox['box'][1][0], bbox['box'][2][0], bbox['box'][3][0])))
            y1 = abs(min((bbox['box'][0][1], bbox['box'][1][1], bbox['box'][2][1], bbox['box'][3][1])))
            x2 = abs(max((bbox['box'][0][0], bbox['box'][1][0], bbox['box'][2][0], bbox['box'][3][0])))
            y2 = abs(max((bbox['box'][0][1], bbox['box'][1][1], bbox['box'][2][1], bbox['box'][3][1])))

            x1, x2 = x1 / image.shape[1], x2 / image.shape[1]
            y1, y2 = y1 / image.shape[0], y2 / image.shape[0]

            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)

            text = bbox['text']
            bboxes.append([x, y, w, h, text])

        if self.transforms:
            augmentations = self.transforms(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        targets = [torch.zeros(self.anchors_per_scale, scale, scale, 5) for scale in self.scales]
        for bbox in bboxes:
            iou_ids = intersection_over_union(torch.tensor(bbox[2:4]), self.anchors, mode='width-height')

            x, y, w, h, _ = bbox
            anchor_ids = torch.argsort(iou_ids, dim=0, descending=True)
            scales_taken = [False, False, False]

            for anchor_id in anchor_ids:
                scale_id = anchor_id // self.anchors_per_scale
                on_scale_id = anchor_id % self.anchors_per_scale
                scale = self.scales[scale_id]

                i, j = int(y*scale), int(x*scale)

                if targets[scale_id][on_scale_id, i, j, 0] == 1:
                    continue

                elif scales_taken[scale_id]:
                    if iou_ids[anchor_id] >= self.ignore_threshold:
                        targets[scale_id][on_scale_id, i, j, 0] = -1
                    continue

                scales_taken[scale_id] = True
                cell_y, cell_x = y*scale - i, x*scale - j
                cell_w, cell_h = w*scale, h*scale
                target = [1, cell_x, cell_y, cell_w, cell_h]

                targets[scale_id][on_scale_id, i, j, 0:6] = torch.tensor(target)

        return image, targets
