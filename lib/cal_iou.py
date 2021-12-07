from shapely.geometry import Polygon


def cal_iou(pol1_xy, pol2_xy):
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    # Calculate Intersection and union, and tne IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.union(polygon2_shape).area
    iou = polygon_intersection / polygon_union
    return iou
