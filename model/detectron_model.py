from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

from lib.data_transform import xylist_to_xyxy


class Dtctron_model:
    def __init__(self, cfg_file):
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        cfg.MODEL.DEVICE = "cpu"

        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)

        self.categories = MetadataCatalog.get(
            cfg.DATASETS.TRAIN[0]).thing_classes

        self.setting_area = []

    def update_setting_area(self, setting_area):
        self.setting_area = xylist_to_xyxy(setting_area)

    def get_predict_reuslt(self, img):
        prediction = self.predictor(img)

        instances = prediction["instances"]
        scores = instances.get_fields()["scores"].tolist()
        pred_classes = instances.get_fields()["pred_classes"].tolist()
        pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()

        scores, pred_classes, pred_boxes = self._only_car(
            scores, pred_classes, pred_boxes, img)

        v = Visualizer(img[:, :, ::-1],
                       MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(instances.to("cpu"))
        cv2.imwrite('static/detect.jpg', out.get_image()[:, :, ::-1])

        return scores, pred_classes, pred_boxes

    def _only_car(self, scores, pred_classes, pred_boxes, image):
        only_scores = []
        only_pred_classes = []
        only_pred_boxes = []
        for i, v in enumerate(pred_classes):
            if self.categories[v] == "car":
                only_scores.append(scores[i])
                only_pred_classes.append(v)
                only_pred_boxes.append(pred_boxes[i])

                box = pred_boxes[i]
                box = [int(i) for i in box]
                top_left, bottom_right = box[:2], box[2:]
                image = cv2.rectangle(
                    image, tuple(top_left), tuple(
                        bottom_right), (0, 255, 0), 1
                )

        cv2.imwrite('static/detect_only_car.jpg', image)

        return only_scores, only_pred_classes, only_pred_boxes

    def overlay_setting_area(self, image):
        for area in self.setting_area:
            top_left, bottom_right = area[0], area[1]

            image = cv2.rectangle(
                image, tuple(top_left), tuple(
                    bottom_right), (255, 0, 0), 1
            )
        return image
