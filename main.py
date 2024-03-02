import sys
import numpy as np
sys.path.insert(0, r"C:\Programming\Projects\LicenseNumber")
sys.path.insert(1, r"C:\Programming\Projects\LicenseNumber\LicenseNumberLocalization")
sys.path.insert(2, r"C:\Programming\Projects\LicenseNumber\OCR")
sys.path.insert(3, r"C:\Programming\Projects\LicenseNumber\localizationAndOCR")

import modelYoloV3
import simpleOCR

import utilsYolo
import utilsOCR
import utils

import configOCR
import configYolo
import configAPI
import configConnection

import torch

import os
from flask import Flask, request, jsonify
from PIL import Image


# import matplotlib.pyplot as plt
# import matplotlib.patches as patches


def model_connection(modelOCR, modelYolo, image_path):
    orig_img = Image.open(image_path).convert("RGB")
    augmentations = configYolo.connect_transforms(image=np.array(orig_img))
    img = augmentations['image']
    img = img.unsqueeze(0).to(configConnection.DEVICE)

    result = ""
    predictions = modelYolo(img)
    bboxes = []
    for prediction in predictions:
        bboxes += utilsYolo.bboxes_conversion(prediction, prediction.shape[3])[0]
    bboxes = utils.to_corners_conversion(bboxes)
    bboxes = utilsYolo.non_max_suppression(
        bboxes,
        configYolo.NMS_IOU_THRESHOLD,
        configYolo.NMS_PROB_THRESHOLD
    )
    for idx, bbox in enumerate(bboxes):
        if orig_img.size[0] > orig_img.size[1]:
            bbox = [
                bbox[1] * orig_img.size[0],
                bbox[2] * orig_img.size[0] - (orig_img.size[0] - orig_img.size[1]) // 2,
                bbox[3] * orig_img.size[0],
                bbox[4] * orig_img.size[0] - (orig_img.size[0] - orig_img.size[1]) // 2
            ]
        else:
            bbox = [
                bbox[1] * orig_img.size[1] - (orig_img.size[1] - orig_img.size[0]) // 2,
                bbox[2] * orig_img.size[1],
                bbox[3] * orig_img.size[1] - (orig_img.size[1] - orig_img.size[0]) // 2,
                bbox[4] * orig_img.size[1]
            ]
        crop_img = orig_img.crop(bbox).convert("L")
        crop_img = np.array(crop_img)
        crop_img = configOCR.TRANSFORMS(image=crop_img)['image']
        crop_img = crop_img.unsqueeze(0).to(configConnection.DEVICE)

        number = modelOCR(crop_img)

        result += utilsOCR.return_result(number) + "\n"

    return result


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in configAPI.ALLOWED_EXTENSIONS


def main():
    modelYolo = modelYoloV3.YoloV3()
    utilsYolo.load_checkpoint(modelYolo, torch.optim.Adam(modelYolo.parameters()))
    modelYolo.eval()
    modelYolo = modelYolo.to(configConnection.DEVICE)

    modelOCR = simpleOCR.SimpleOCR()
    utilsOCR.load_model_checkpoint(modelOCR, torch.optim.Adam(modelOCR.parameters()))
    modelOCR.eval()
    modelOCR = modelOCR.to(configConnection.DEVICE)

    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = configAPI.UPLOAD_FOLDER

    @app.route('/', methods=['GET', 'POST'])
    def upload_file():
        if request.method == 'POST':
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                return model_connection(modelOCR, modelYolo, file_path), 201
            else:
                return "Unsupported Media Type", 415

    app.run()


if __name__ == "__main__":
    main()

