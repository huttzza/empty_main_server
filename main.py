# reference from https://gist.github.com/ylashin/9911cea4a42f8b18f74bafa1952379a5

from flask import Flask
from flask_cors import CORS
from flask import request, jsonify, abort, Response
from werkzeug.wrappers import response
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import shutil
import cv2
import requests
import numpy as np
import threading
import json
from constant import CFG_FILE, PI_IP


def score_image(predictor: DefaultPredictor, image_url: str, mode: str):
    if mode == 'url':
        image_reponse = requests.get(image_url)
        image_as_np_array = np.frombuffer(image_reponse.content, np.uint8)
    elif mode == 'local':
        with open('static/parking_lot.jpg', 'rb') as f:
            image_local = f.read()
        image_as_np_array = np.frombuffer(image_local, np.uint8)
    else:
        abort(405)

    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)

    # make prediction
    return image, predictor(image)


def prepare_pridctor():
    # create config
    cfg = get_cfg()
    cfg.merge_from_file(CFG_FILE)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.MODEL.DEVICE = "cpu"  # we use a CPU Detectron copy

    classes = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes
    predictor = DefaultPredictor(cfg)
    print("Predictor has been initialized.")
    return (predictor, classes, cfg)


def get_pklot():
    response = requests.get(PI_IP, stream=True)
    with open('static/parking_lot.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    print("parking lot image get from pi-camera")

    # get pklot image from pi camera every 5 seconds
    threading.Timer(30, get_pklot).start()


app = Flask(__name__)
CORS(app)
predictor, classes, cfg = prepare_pridctor()
get_pklot()


@app.route("/predict", methods=["POST"])
def process_score_image_request():
    image_url = request.json["imageUrl"]
    mode = request.json["mode"]
    im, scoring_result = score_image(predictor, image_url, mode)

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
        # "classes": classes
    }

    return jsonify(response)


@app.route("/test")
def test_json():
    response = [{"isEmpty": True, "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2}, {"isEmpty": False, "x1": 0.2, "y1": 0.1, "x2": 0.3, "y2": 0.3}, {"isEmpty": True, "x1": 0.3, "y1": 0.2, "x2": 0.4, "y2": 0.4}, {"isEmpty": True, "x1": 0.4, "y1": 0.3, "x2": 0.5, "y2": 0.5}, {"isEmpty": False, "x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.6}, {
        "isEmpty": False, "x1": 0.7, "y1": 0.1, "x2": 0.8, "y2": 0.2}, {"isEmpty": False, "x1": 0.8, "y1": 0.1, "x2": 0.9, "y2": 0.3}, {"isEmpty": False, "x1": 0.9, "y1": 0.2, "x2": 1.0, "y2": 0.4}, {"isEmpty": True, "x1": 1.0, "y1": 0.3, "x2": 1.0, "y2": 0.5}, {"isEmpty": False, "x1": 1.0, "y1": 0.5, "x2": 1.0, "y2": 0.6}]

    return Response(json.dumps(response), mimetype="application/json", status=200)


@app.route("/images")
def show_pklot():
    return "<img src='static/parking_lot.jpg'> <br> <img src='static/detect.jpg'>"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80, debug=True)
