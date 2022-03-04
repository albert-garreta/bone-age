import cv2


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
    a = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = point1[1] - a * point1[0]

    def line_function(x):
        return a * x + b

    return line_function
