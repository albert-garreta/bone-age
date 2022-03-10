import cv2
import numpy as np


def annotate_img(
    _img,
    point,
    _annotation,
    _font_face=cv2.FONT_HERSHEY_SIMPLEX,
    _font_scale=2,
    _font_color=(255, 0, 255),
    _thickness=2,
):
    cv2.putText(
        _img,
        _annotation,
        point,
        _font_face,
        _font_scale,
        _font_color,
        _thickness,
    )


def get_line_function(point1, point2):
    if not point1 or not point2:
        return None
    a = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - a * point1[0]

    def line_function(x):
        return a * x + b

    return line_function


def get_inverse_perp_line(point1, point2):
    if not point1 or not point2:
        return None
    a = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - a * point1[0]

    mid_point = (0.5 * (point1[0] + point2[0]), 0.5 * (point1[1] + point2[1]))
    b = mid_point[1] + (1/a)*mid_point[0]
    
    def line_function(y):
        return (y-b)*(-a) 

    return line_function


def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
