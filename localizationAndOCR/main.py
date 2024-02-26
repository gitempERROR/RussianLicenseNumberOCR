from LicenseNumberLocalization import utilsYolo
from LicenseNumberLocalization import configYolo
from LicenseNumberLocalization import modelYoloV3
from OCR import utilsOCR
from OCR import simpleOCR
from OCR import configOCR
from PIL import Image
from PIL import ImageEnhance
from localizationAndOCR import configConnection
from localizationAndOCR import utils
import torch
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def model_connection(modelOCR, modelYolo):
    orig_img = Image.open(r"C:\Programming\Projects\LicenseNumber\33.jpg").convert("RGB")
    img = configYolo.connect_transforms(image=np.array(orig_img))['image']
    img = img.unsqueeze(0).to(configConnection.DEVICE)
    preds = modelYolo(img)
    bboxes = []
    for pred in preds:
        bboxes += utilsYolo.bboxes_conversion(pred, pred.shape[3])[0]
    bboxes = utils.to_corners_conversion(bboxes)
    bboxes = utilsYolo.non_max_suppression(
        bboxes,
        configYolo.NMS_IOU_THRESHOLD,
        configYolo.NMS_PROB_THRESHOLD
    )
    for idx, bbox in enumerate(bboxes):
        if orig_img.size[0] > orig_img.size[1]:
            bbox = [
                bbox[1]*orig_img.size[0],
                bbox[2]*orig_img.size[0] - (orig_img.size[0] - orig_img.size[1]) // 2,
                bbox[3]*orig_img.size[0],
                bbox[4]*orig_img.size[0] - (orig_img.size[0] - orig_img.size[1]) // 2
            ]
        else:
            bbox = [
                bbox[1]*orig_img.size[1] - (orig_img.size[1] - orig_img.size[0]) // 2,
                bbox[2]*orig_img.size[1],
                bbox[3]*orig_img.size[1] - (orig_img.size[1] - orig_img.size[0]) // 2,
                bbox[4]*orig_img.size[1]
            ]
        # plt.imshow(orig_img)
        # rect = patches.Rectangle(
        #     (bbox[0], bbox[1]),
        #     bbox[2]-bbox[0],
        #     bbox[3]-bbox[1],
        #     linewidth=2,
        #     edgecolor="red",
        #     facecolor="none",
        #  )
        # # Add the patch to the Axes
        # plt.gca().add_patch(rect)
        # plt.show()
        crop_img = orig_img.crop(bbox).convert("L")
        crop_img.save(f"test_img{idx}.jpg")
        crop_img = np.array(crop_img)
        crop_img = configOCR.TRANSFORMS(image=crop_img)['image']
        crop_img = crop_img.unsqueeze(0).to(configConnection.DEVICE)
        number = modelOCR(crop_img)
        utilsOCR.print_result(number)


if __name__ == "__main__":
    modelYolo = modelYoloV3.YoloV3()
    utilsYolo.load_checkpoint(modelYolo, torch.optim.Adam(modelYolo.parameters()))
    modelYolo.eval()
    modelYolo = modelYolo.to(configConnection.DEVICE)

    modelOCR = simpleOCR.SimpleOCR()
    utilsOCR.load_model_checkpoint(modelOCR, torch.optim.Adam(modelOCR.parameters()))
    modelOCR.eval()
    modelOCR = modelOCR.to(configConnection.DEVICE)

    model_connection(modelOCR, modelYolo)


