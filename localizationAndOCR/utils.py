def to_corners_conversion(bbox_predictions):
    for i in range(len(bbox_predictions)):
        bbox = bbox_predictions[i]
        x1 = bbox[1] - bbox[3] / 2
        x2 = bbox[1] + bbox[3] / 2
        y1 = bbox[2] - bbox[4] / 2
        y2 = bbox[2] + bbox[4] / 2
        bbox[1:5] = (x1, y1, x2, y2)
        bbox_predictions[i] = bbox
    return bbox_predictions

