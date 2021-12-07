import base64
import numpy as np
import cv2


def xyxy_to_xylist(xyxy_points):
    # top left xy / bottom right xy
    for i, xyxy in enumerate(xyxy_points):
        x1, y1, x2, y2 = map(int, xyxy)
        xylist = [[x1, y1], [x2, y1], [x2, y1], [x2, y2]]
        xyxy_points[i] = xylist
    return xyxy_points


def xylist_to_center(xyxy_points):
    for i, xy in enumerate(xyxy_points):
        x_list = [point[0] for point in xy]
        y_list = [point[1] for point in xy]

        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        x = abs(x_max - x_min)
        y = abs(y_max - y_min)

        xyxy_points[i] = [x, y]
    return xyxy_points


def decode_base64(data):
    # base64string -> base64Image
    imageStr = base64.b64decode(data)
    nparr = np.fromstring(imageStr, np.uint8)
    base64Image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return base64Image


def encode_base64(image_path):
    # base64Image -> base64string
    with open(image_path, 'rb') as img:
        base64Image = base64.b64encode(img.read()).decode('utf8')
    return base64Image
