from maskrcnn_benchmark.config import cfg
from model.predictor import *


class Mskrcnn_model:
    def __init__(self, cfg_file, model_file, categories):
        self.categories = categories

        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHT = model_file
        cfg.MODEL.DEVICE = "cpu"

        self.cfg = cfg
        self.predictor = Predictor(cfg, categories=categories)

        # return predictor, cfg

    def get_predict_reuslt(self, img):
        result_img, prediction = self.predictor.run_on_opencv_image(img)

        cv2.imwrite('static/detect.jpg', result_img)

        scores = prediction.get_field("scores").tolist()
        pred_classes = prediction.get_field("labels").tolist()
        pred_boxes = prediction.bbox.tolist()

        scores, pred_classes, pred_boxes = self._only_car(
            scores, pred_classes, pred_boxes)

        return scores, pred_classes, pred_boxes

    def _only_car(self, scores, pred_classes, pred_boxes):
        only_scores = []
        only_pred_classes = []
        only_pred_boxes = []
        for i, v in enumerate(pred_classes):
            if v == 3:  # car
                only_scores.append(scores[i])
                only_pred_classes.append(v)
                only_pred_boxes.append(pred_boxes[i])
        return only_scores, only_pred_classes, only_pred_boxes
