from flask import Flask
from flask_cors import CORS
from flask import request, abort, Response

import shutil
import cv2
import requests
import numpy as np
import threading
import json
import argparse
from detectron2.data import MetadataCatalog

from lib.constant import *
from lib.category import *
from lib.maskrcnn_model import *
from lib.detectron_model import *
from lib.yaml_path import *


app = Flask(__name__)
CORS(app)


def read_image(image_url, mode):
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

    return image


def get_label_name(labels, categories):
    label_name = []

    for label in labels:
        if label >= len(categories):
            abort(405)
        label_name.append(categories[label])
    return label_name


def get_pklot_from_pi():
    try:
        response = requests.get(PI_IP, stream=True)
        with open('static/parking_lot.jpg', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        print("parking lot image get from pi-camera")

        # get pklot image from pi camera every 5 seconds
        threading.Timer(30, get_pklot_from_pi).start()
    except Exception as e:
        print(e)


# -----------------------flask--------------------------
@ app.route("/predict", methods=["POST"])
def process_score_image_request():
    image_path = request.json["imageUrl"]
    mode = request.json["mode"]
    image = read_image(image_path, mode)

    width, height, _ = image.shape

    scores, pred_classes, pred_boxes = predictor.get_predict_reuslt(image)

    pred_class_name = get_label_name(pred_classes, predictor.categories)

    response = {
        # "scores": scores,
        # "pred_classes": pred_classes,
        "pred_class_name": pred_class_name,
        "pred_boxes": pred_boxes
    }

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
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--detectron', '-d',
                       dest='isDetectron', action='store_true')
    group.add_argument('--maskrcnn', '-m',
                       dest='isDetectron', action='store_false')
    parser.set_defaults(isDetectron=True)
    args = parser.parse_args()

    model_path = MODEL_PATH
    categories = CATEGORIES

    if args.isDetectron:
        print("run on detectron")
        cfg_file = FASTER_RCNN
        predictor = Dtctron_model(cfg_file)
    else:
        print("run on maskrcnn benchmark")
        cfg_file = DA_FASTER_RCNN
        predictor = Mskrcnn_model(cfg_file, model_path, categories)

    get_pklot_from_pi()

    app.run(host="0.0.0.0", port=4000, debug=True)
