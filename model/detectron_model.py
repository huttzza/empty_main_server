from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2


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

        # return predictor, cfg

    def get_predict_reuslt(self, img):
        prediction = self.predictor(img)

        instances = prediction["instances"]
        scores = instances.get_fields()["scores"].tolist()
        pred_classes = instances.get_fields()["pred_classes"].tolist()
        pred_boxes = instances.get_fields()["pred_boxes"].tensor.tolist()

        scores, pred_classes, pred_boxes = self._only_car(
            scores, pred_classes, pred_boxes)

        v = Visualizer(img[:, :, ::-1],
                       MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(instances.to("cpu"))
        cv2.imwrite('static/detect.jpg', out.get_image()[:, :, ::-1])

        return scores, pred_classes, pred_boxes

    def _only_car(self, scores, pred_classes, pred_boxes):
        only_scores = []
        only_pred_classes = []
        only_pred_boxes = []
        for i, v in enumerate(pred_classes):
            if v == 2:  # car
                only_scores.append(scores[i])
                only_pred_classes.append(v)
                only_pred_boxes.append(pred_boxes[i])
        return only_scores, only_pred_classes, only_pred_boxes
