import os
from flask import Flask
from flask_cors import CORS
from flask import request, abort, Response, send_file

import shutil
import copy
import cv2
import requests
import numpy as np
import threading
import json
import argparse
from detectron2.data import MetadataCatalog
from lib.cal_iou import cal_iou

from config.constant import *
from config.category import *
from model.maskrcnn_model import *
from model.detectron_model import *
from lib.data_transform import encode_base64, xylist_to_center, xyxy_to_xylist
from config.yaml_path import *


app = Flask(__name__)
CORS(app)
response_is_full = []
table = []


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


def process_detection(image):
    width, height, _ = image.shape

    scores, pred_classes, pred_boxes = predictor.get_predict_reuslt(image)

    pred_class_name = get_label_name(pred_classes, predictor.categories)
    pred_boxes = xyxy_to_xylist(pred_boxes)

    return scores, pred_classes, pred_class_name, pred_boxes


def detect_and_cal_iou(setting_area):
    global response_is_full

    # TODO 영상 처음부터 끝까지 읽어오기
    filePath = VIDEO_PATH
    print(filePath)

    if os.path.isfile(filePath):  # 해당 파일이 있는지 확인
        # 영상 객체(파일) 가져오기
        capture = cv2.VideoCapture(filePath)
    else:
        print("file does not exist")
        return

    FPS = int(capture.get(cv2.CAP_PROP_FPS))
    count = FPS

    while True:
        retval, frame = capture.read()

        if not retval:  # 프레임 정보 정상적으로 읽지 못했을 때
            break
        #cv2.imshow('frame', frame)

        count += 1
        if count < FPS:
            continue
        count = 0

        _, _, _, detect_from_video = process_detection(frame)

        is_full = [False for i in range(len(setting_area))]
        true_count = 0
        for i, area in enumerate(setting_area):
            for detect in detect_from_video:
                iou = cal_iou(area, detect)

                if iou > 0.6:
                    # print(iou)
                    is_full[i] = True
                    true_count += 1
                    break
        response_is_full = is_full
        # print(response_is_full)
        print("found " + str(true_count) + " cars")
        print("is_full updated")

    capture.release()
    cv2.destroyAllWindows()
    print("video detection finished")

# -----------------------flask--------------------------


@ app.route("/", methods=["GET"])
def pi_image_for_first_setting():
    first_setting_img = 'static/parking_lot.png'

    with open(first_setting_img, 'rb') as f:
        image_local = f.read()
    image_as_np_array = np.frombuffer(image_local, np.uint8)
    image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)

    scores, pred_classes, pred_class_name, pred_boxes = process_detection(
        image)
    response = {
        # "scores": scores,
        # "pred_classes": pred_classes,
        # "pred_class_name": pred_class_name,
        "pred_boxes": pred_boxes
    }
    response["image"] = encode_base64(first_setting_img)

    return json.dumps(response)


@ app.route("/set", methods=["POST"])
def set_and_run():
    global table
    table = request.json["table"]
    xyxy_points = request.json["setting_area"]
    print(str(len(xyxy_points)) + " setting area")

    predictor.update_setting_area(copy.deepcopy(xyxy_points))

    threading.Thread(target=detect_and_cal_iou, args=[xyxy_points]).start()
    print("start video detection ... ")
    return Response()


@ app.route("/video_predict", methods=["GET"])
def vidoe_detection():
    global response_is_full
    _response_is_full = []
    for i, is_full in enumerate(response_is_full):
        point = {"label": i, "is_full": is_full}
        _response_is_full.append(point)

    return {"table": table, "area": _response_is_full}


@ app.route("/predict", methods=["POST"])
def process_detection_request():
    image_path = request.json["imageUrl"]
    mode = request.json["mode"]
    image = read_image(image_path, mode)

    width, height, _ = image.shape

    scores, pred_classes, pred_boxes = predictor.get_predict_reuslt(image)

    pred_class_name = get_label_name(pred_classes, predictor.categories)
    pred_boxes = xyxy_to_xylist(pred_boxes)

    response = {
        # "scores": scores,
        # "pred_classes": pred_classes,
        # "pred_class_name": pred_class_name,
        "pred_boxes": pred_boxes
    }

    return response


@ app.route("/test")
def test_json():
    response = {
        "area": [
            {
                "is_full": True,
                "label": 0
            },
            {
                "is_full": True,
                "label": 1
            }
        ],
        "table": [
            1,
            1
        ]
    }

    return response


@ app.route("/images")
def show_pklot():
    return "<img src='static/parking_lot.png'> <br> <img src='static/detect.jpg'>"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--detectron', '-d',
                       dest='isDetectron', action='store_true')
    group.add_argument('--maskrcnn', '-m',
                       dest='isDetectron', action='store_false')
    parser.set_defaults(isDetectron=False)
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

    app.run(host="0.0.0.0", port=8000, debug=True)
