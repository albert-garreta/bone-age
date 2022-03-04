from math import sin, cos
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.linalg import norm
import numpy as np
from math import atan
import os
from scipy.spatial import procrustes

def get_translation(shape):
    """
    Calculates a translation for x and y
    axis that centers shape around the
    origin
    Args:
      shape(2n x 1 NumPy array) an array
      containing x coodrinates of shape
      points as first column and y coords
      as second column
     Returns:
      translation([x,y]) a NumPy array with
      x and y translationcoordinates
    """

    mean_x = int(np.mean(shape[::2]))
    mean_y = int(np.mean(shape[1::2]))

    return np.array([mean_x, mean_y])


def translate(shape):
    """
    Translates shape to the origin
    Args:
      shape(2n x 1 NumPy array) an array
      containing x coodrinates of shape
      points as first column and y coords
      as second column
    """
    mean_x, mean_y = get_translation(shape)
    shape[::2] -= mean_x
    shape[1::2] -= mean_y


def get_rotation_scale(reference_shape, shape):
    """
    Calculates rotation and scale
    that would optimally align shape
    with reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference for scaling and
        alignment

        shape(2nx1 NumPy array), a shape that is scaled
        and aligned

    Returns:
        scale(float), a scaling factor
        theta(float), a rotation angle in radians
    """

    a = np.dot(shape, reference_shape) / norm(reference_shape) ** 2

    # separate x and y for the sake of convenience
    ref_x = reference_shape[::2]
    ref_y = reference_shape[1::2]

    x = shape[::2]
    y = shape[1::2]

    b = np.sum(x * ref_y - ref_x * y) / norm(reference_shape) ** 2

    scale = np.sqrt(a ** 2 + b ** 2)
    theta = atan(b / max(a, 10 ** -10))  # avoid dividing by 0

    return round(scale, 1), round(theta, 2)


def get_rotation_matrix(theta):

    return np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])


def scale(shape, scale):

    return shape / scale


def rotate(shape, theta):
    """
    Rotates a shape by angle theta
    Assumes a shape is centered around
    origin
    Args:
        shape(2nx1 NumPy array) an shape to be rotated
        theta(float) angle in radians
    Returns:
        rotated_shape(2nx1 NumPy array) a rotated shape
    """

    matr = get_rotation_matrix(theta)

    # reshape so that dot product is eascily computed
    temp_shape = shape.reshape((-1, 2)).T

    # rotate
    rotated_shape = np.dot(matr, temp_shape)

    return rotated_shape.T.reshape(-1)

resize_factor = 1
def prepare(img_arrays):
    max_width = max([h.shape[1] for h in img_arrays])
    max_height = max( [h.shape[0] for h in img_arrays])

    for idx, h in enumerate(img_arrays):
        h = cv2.cvtColor(h, cv2.COLOR_RGB2GRAY)
        h = cv2.equalizeHist(h)
        extra_zero_rows = np.zeros((max_height - h.shape[0], h.shape[1]))
        h = np.concatenate([h, extra_zero_rows], axis=0)
        extra_zero_cols = np.zeros((h.shape[0], max_width -h.shape[1]))
        h = np.concatenate([h, extra_zero_cols], axis=1)
        assert h.shape == ( max_height, max_width)
        print(max_height, max_width)
        h = cv2.resize(h, (int(resize_factor*max_width), int(resize_factor*max_height)),interpolation = cv2.INTER_AREA)
        # plt.imshow(h)
        # plt.show()
        img_arrays[idx] = h.flatten()

    return np.stack(img_arrays)


def procrustes_analysis(reference_shape, shape):
    """
    Scales, and rotates a shape optimally to
    be aligned with a reference shape
    Args:
        reference_shape(2nx1 NumPy array), a shape that
        serves as reference alignment

        shape(2nx1 NumPy array), a shape that is aligned

    Returns:
        aligned_shape(2nx1 NumPy array), an aligned shape
        translated to the location of reference shape
    """
    # copy both shapes in caseoriginals are needed later
    temp_ref = np.copy(reference_shape)
    temp_sh = np.copy(shape)

    translate(temp_ref)
    translate(temp_sh)

    # get scale and rotation
    scale, theta = get_rotation_scale(temp_ref, temp_sh)
    print("scale and rotation", scale, theta)
    # scale, rotate both shapes
    temp_sh = temp_sh / scale
    aligned_shape = rotate(temp_sh, theta)

    return aligned_shape


if __name__ == '__main__':
    dir = "data/data_tagged"

    img1_dir = os.path.join(dir, "1377.png")
    img2_dir = os.path.join(dir, "13649.png")

    img1 = cv2.imread(img1_dir)
    img2 = cv2.imread(img2_dir)
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()

    arrays = prepare([img1, img2])
    print(arrays.shape)
    im1, im2 = arrays[[0]], arrays[[1]]
    print(im1.shape, im2.shape)
    result  = procrustes(im1, im2)
    #result = procrustes_analysis(arrays[0], arrays[1])

    print(result.shape)

    result  =result.reshape(2570, 2040)
    plt.imshow(result)
    plt.show()
    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    