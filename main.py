from flask import Flask
from flask_cors import CORS
from flask import request, abort, Response
from maskrcnn_benchmark.config import cfg
import shutil
import cv2
import requests
import numpy as np
import threading
import json
from lib.constant import *
from lib.category import *
from lib.predictor import *

app = Flask(__name__)
CORS(app)


def initial_setting(cfg_file, model_file, categories):
    cfg.merge_from_file(cfg_file)
    cfg.MODEL.WEIGHT = model_file
    cfg.MODEL.DEVICE = "cpu"

    predictor = Predictor(cfg, categories=categories)

    return predictor


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

    result_img, prediction = predictor.run_on_opencv_image(image)

    cv2.imwrite('static/detect.jpg', result_img)
    print("successfully predict end")

    scores = prediction.get_field("scores").tolist()
    labels = prediction.get_field("labels").tolist()
    labels_name = get_label_name(labels, categories)
    bbox = prediction.bbox.tolist()

    response = {
        # "scores": scores,
        # "labels": labels,
        "labels_name": labels_name,
        "bbox": bbox
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
    cfg_file = CFG_FILE
    model_path = MODEL_PATH
    categories = CATEGORIES

    predictor = initial_setting(cfg_file, model_path, categories)
    get_pklot_from_pi()

    app.run(host="0.0.0.0", port=4000, debug=True)
