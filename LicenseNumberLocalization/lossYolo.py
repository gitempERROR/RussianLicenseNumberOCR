import torch
import torch.nn as nn

from utilsYolo import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()
        self.exp = torch.exp
        self.sigm = nn.Sigmoid()

        self.lambda_object = 5
        self.lambda_no_object = 10
        self.lambda_box = 5

    def forward(self, predictions, target, anchors) -> torch.tensor:
        """
        :param predictions: предсказания модели
        :param target: целевые рамки
        :param anchors: якорные точки
        :return:
        """
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0

        # no object loss
        no_object_loss = self.bce(predictions[..., 0:1][no_obj], target[..., 0:1][no_obj])

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)

        box_preds = torch.cat([self.sigm(predictions[..., 0:3]), self.exp(predictions[..., 3:5])*anchors], dim=-1)

        iou = intersection_over_union(box_preds[..., 1:5][obj], target[..., 1:5][obj], mode="midpoint").detach()
        object_loss = self.bce(predictions[..., 0:1][obj], target[..., 0:1][obj] * iou)

        # box coordinates loss
        predictions[..., 1:3] = self.sigm(predictions[..., 1:3])
        target[..., 3:5] = torch.log(target[..., 3:5] / anchors + 1e-16)

        box_coordinates_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        total_loss = (
            no_object_loss * self.lambda_no_object
            + object_loss * self.lambda_object
            + box_coordinates_loss * self.lambda_box
        )

        return total_loss
