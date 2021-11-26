# reference from https://gist.github.com/ylashin/9911cea4a42f8b18f74bafa1952379a5

from flask import Flask
from flask_cors import CORS
from flask import request, jsonify, abort, Response
import torch
from detectron2.data import MetadataCatalog
from maskrcnn_benchmark.structures.image_list import to_image_list
import argparse
import shutil
import cv2
import requests
import numpy as np
import threading
import json
from constant import *
from category import *
from cocoDemo_func import *

app = Flask(__name__)
CORS(app)
OPTION = "custom"


def score_image(predictor, image_url, mode):
    if mode == 'url':
        image_response = requests.get(image_url)
        image_as_np_array = np.frombuffer(image_response.content, np.uint8)
    elif mode == 'local':
        with open('static/parking_lot.png', 'rb') as f:
            image_local = f.read()
        image_as_np_array = np.frombuffer(image_local, np.uint8)
    else:
        abort(405)

    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)
    cv2.imwrite('static/image.jpg', image)

    transforms = build_transform()
    original_image = image.copy()
    image = transforms(image)
    image_list = to_image_list(image, cfg.DATALOADER.SIZE_DIVISIBILITY)
    image_list = image_list.to(cfg.MODEL.DEVICE)
    with torch.no_grad():
        predictions = predictor(image_list)

    predictions = [o.to(cfg.MODEL.DEVICE) for o in predictions]
    result = predictions[0]

    height, width = original_image.shape[:-1]
    prediction = predictions[0].resize((width, height))

    result_img = original_image.copy()
    cv2.imwrite('static/original.jpg', original_image)
    result_img = overlay_boxes(result_img, prediction)
    result_img = overlay_class_names(result_img, prediction)

    image = result_img

    return image, result


def prepare_pridctor():
    cfg.merge_from_file(CFG_FILE)
    cfg.MODEL.WEIGHTS = MODEL_PATH
    cfg.MODEL.DEVICE = "cpu"

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)
    checkpointer = DetectronCheckpointer(cfg, model)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)
    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN)
    model.eval()

    return (model, classes, cfg)


def get_pklot():
    try:
        response = requests.get(PI_IP, stream=True)
        with open('static/parking_lot.jpg', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        print("parking lot image get from pi-camera")

        # get pklot image from pi camera every 5 seconds
        threading.Timer(30, get_pklot).start()
    except Exception as e:
        print(e)


@ app.route("/predict", methods=["POST"])
def process_score_image_request():
    image_url = request.json["imageUrl"]
    mode = request.json["mode"]
    im, scoring_result = score_image(predictor, image_url, mode)
    '''
    instances = scoring_result["instances"]
    scores = instances.get_fields()["scores"].tolist()
    pred_classes = instances.get_fields()["pred_classes"].tolist()
    pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()
    pred_class_name = [classes[i] for i in pred_classes]

    v = Visualizer(
        im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(instances.to("cpu"))
    cv2.imwrite('static/detect.jpg', out.get_image()[:, :, ::-1])

    response = {
        "scores": scores,
        "pred_classes": pred_classes,
        "pred_class_name": pred_class_name,
        "pred_boxes": pred_boxes,
        "classes": classes
    }
    '''
    cv2.imwrite('static/detect.jpg', im)

    response = {
        "scores": scoring_result.get_field("scores").tolist(),
        "labels": scoring_result.get_field("labels").tolist(),
        "bbox": scoring_result.bbox.tolist()
    }
    # print(scoring_result.fields())
    # return jsonify(response)
    # print(response)
    return response


@ app.route("/test")
def test_json():
    response = [{"isEmpty": True, "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}, {"isEmpty": False, "x1": 0.2, "y1": 0.1, "x2": 0.3, "y2": 0.3}, {"isEmpty": True, "x1": 0.3, "y1": 0.2, "x2": 0.4, "y2": 0.4}, {"isEmpty": True, "x1": 0.4, "y1": 0.3, "x2": 0.5, "y2": 0.5}, {"isEmpty": False, "x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.6}, {
        "isEmpty": False, "x1": 0.7, "y1": 0.1, "x2": 0.8, "y2": 0.2}, {"isEmpty": False, "x1": 0.8, "y1": 0.1, "x2": 0.9, "y2": 0.3}, {"isEmpty": False, "x1": 0.9, "y1": 0.2, "x2": 1.0, "y2": 0.4}, {"isEmpty": True, "x1": 1.0, "y1": 0.3, "x2": 1.0, "y2": 0.5}, {"isEmpty": False, "x1": 1.0, "y1": 0.5, "x2": 1.0, "y2": 0.6}]

    return Response(json.dumps(response), mimetype="application/json", status=200)


@ app.route("/images")
def show_pklot():
    return "<img src='static/parking_lot.jpg'> <br> <img src='static/detect.jpg'>"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="custom")
    args = parser.parse_args()

    OPTION = args.model  # custom or others

    predictor, classes, cfg = prepare_pridctor()
    get_pklot()

    app.run(host="0.0.0.0", port=80, debug=True)
