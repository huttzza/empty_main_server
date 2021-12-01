from maskrcnn_benchmark.config import cfg
from lib.predictor import *


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

        return scores, pred_classes, pred_boxes
