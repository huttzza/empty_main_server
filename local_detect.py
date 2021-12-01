import argparse
from main import read_image, get_label_name
from lib.constant import *
from lib.maskrcnn_model import *
from lib.detectron_model import *
from lib.yaml_path import *

model_path = MODEL_PATH  # ".pth"
categories = CATEGORIES  # list
image_path = "static/pklots.png"

# -----------------------------------------------

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=False)
group.add_argument('--detectron', '-d',
                   dest='isDetectron', action='store_true')
group.add_argument('--maskrcnn', '-m',
                   dest='isDetectron', action='store_false')
parser.set_defaults(isDetectron=True)
args = parser.parse_args()

if args.isDetectron:
    print("run on detectron")
    cfg_file = FASTER_RCNN
    predictor = Dtctron_model(cfg_file)
else:
    print("run on maskrcnn benchmark")
    cfg_file = DA_FASTER_RCNN
    predictor = Mskrcnn_model(cfg_file, model_path, categories)

with open('static/parking_lot.png', 'rb') as f:
    image_local = f.read()
image_as_np_array = np.frombuffer(image_local, np.uint8)
image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)

scores, pred_classes, pred_boxes = predictor.get_predict_reuslt(image)

pred_class_name = get_label_name(pred_classes, predictor.categories)

response = {
    # "scores": scores,
    # "pred_classes": pred_classes,
    "pred_class_name": pred_class_name,
    "pred_boxes": pred_boxes
}

# print(response)
