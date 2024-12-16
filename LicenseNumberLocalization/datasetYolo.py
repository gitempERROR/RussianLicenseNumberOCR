import torch
import numpy as np
import os
import json
# import xml.etree.ElementTree as ET

from PIL import Image
from utilsYolo import intersection_over_union


class YoloDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, label_file, anchors, transforms=None, scales=(13, 26, 52)):
        """
        :param dataset_dir: string - путь к датасету с изображениями
        :param label_file: string - путь к файлу с разметкой
        :param anchors: list - якорные точки
        :param transforms: albumentations.compose.Compose - преобразования изображений
        :param scales: list<int> - масштабы матриц результата (разные матрицы для разных размеров объектов)
        """
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
            # Получение угловых координат рамок объектов
            x1 = abs(min((bbox['box'][0][0], bbox['box'][1][0], bbox['box'][2][0], bbox['box'][3][0])))
            y1 = abs(min((bbox['box'][0][1], bbox['box'][1][1], bbox['box'][2][1], bbox['box'][3][1])))
            x2 = abs(max((bbox['box'][0][0], bbox['box'][1][0], bbox['box'][2][0], bbox['box'][3][0])))
            y2 = abs(max((bbox['box'][0][1], bbox['box'][1][1], bbox['box'][2][1], bbox['box'][3][1])))

            # Приведение координат к формату от 0 до 1
            x1, x2 = x1 / image.shape[1], x2 / image.shape[1]
            y1, y2 = y1 / image.shape[0], y2 / image.shape[0]

            # Получение середины рамок и их ширины и высоты
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            w = (x2 - x1)
            h = (y2 - y1)

            # Получение подписи рамки, но дальше не используется
            text = bbox['text']
            bboxes.append([x, y, w, h, text])

        if self.transforms:
            # Применение аугментации изображений
            augmentations = self.transforms(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']

        targets = [torch.zeros(self.anchors_per_scale, scale, scale, 5) for scale in self.scales]
        for bbox in bboxes:
            # Расчет IOU относительно якорных точек
            iou_ids = intersection_over_union(torch.tensor(bbox[2:4]), self.anchors, mode='width-height')

            x, y, w, h, _ = bbox
            # Сортировка якорных точек по IOU
            anchor_ids = torch.argsort(iou_ids, dim=0, descending=True)
            # На каждом масштабе только одна рамка
            scales_taken = [False, False, False]

            # Определяем применима ли якорная точка к рамке и на каком масштабе
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
