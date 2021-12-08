from cv2 import threshold
from maskrcnn_benchmark.config import cfg
from numpy import double, float32, heaviside
from lib.data_transform import xylist_to_xyxy
from model.predictor import *


class Mskrcnn_model:
    def __init__(self, cfg_file, model_file, categories):
        self.categories = categories

        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHT = model_file
        cfg.MODEL.DEVICE = "cpu"

        self.cfg = cfg
        self.predictor = Predictor(cfg, categories=categories)

        self.setting_area = []
        #self.threshold = threshold

    def update_setting_area(self, setting_area):
        self.setting_area = xylist_to_xyxy(setting_area)

    def get_predict_reuslt(self, img):
        result_img, prediction = self.predictor.run_on_opencv_image(img)

        img = self.overlay_setting_area(img)
        cv2.imwrite('static/detect.jpg', result_img)

        scores = prediction.get_field("scores").tolist()
        pred_classes = prediction.get_field("labels").tolist()
        pred_boxes = prediction.bbox.tolist()

        scores, pred_classes, pred_boxes = self._only_car(
            scores, pred_classes, pred_boxes, img)

        return scores, pred_classes, pred_boxes

    def _only_car(self, scores, pred_classes, pred_boxes, image):
        only_scores = []
        only_pred_classes = []
        only_pred_boxes = []

        # img_width, img_height, _ = image.shape
        # image_area = img_width * img_height

        for i, v in enumerate(pred_classes):
            box = pred_boxes[i]
            box = [int(i) for i in box]
            top_left, bottom_right = box[:2], box[2:]

            # width = abs(top_left[0] - bottom_right[0])
            # height = abs(top_left[1] - bottom_right[1])
            # box_area = width * height

            # if box_area > self.threshold * image_area and
            if self.categories[v] == "car":
                only_scores.append(scores[i])
                only_pred_classes.append(v)
                only_pred_boxes.append(pred_boxes[i])

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
