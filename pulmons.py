from dataclasses import dataclass
import numpy as np
import os
import pydicom
import matplotlib.pyplot as plt
import cv2
from skimage import measure
from skimage.morphology import convex_hull_image
import re


@dataclass
class Placa:
    PatientID: int
    PatientOrientation: int
    AcquisitionDate: int
    ContentDate: int
    AcquisitionTime: int
    ContentTime: int
    BodyPartExamined: int
    PhotometricInterpretation: str
    PixelArray: np.ndarray
    KVP: float = 0.0
    ExposureInuAs: int = 0
    RelativeXRayExposure: str = None


def create_placa_from_IMG(A):
    d = {
        "PatientID": A.PatientID,
        "PatientOrientation": A.PatientOrientation,
        "AcquisitionDate": A.AcquisitionDate,
        "ContentDate": A.ContentDate,
        "AcquisitionTime": A.AcquisitionTime,
        "ContentTime": A.ContentTime,
        "BodyPartExamined": A.BodyPartExamined,
        "PhotometricInterpretation": A.PhotometricInterpretation,
        "PixelArray": A.pixel_array,
    }
    count_exceptions1 = 0
    count_exceptions2 = 0
    count_exceptions3 = 0
    try:
        d["KVP"] = A.KVP
    except:
        count_exceptions1 += 1
    try:
        d["ExposureInuAs"] = A.ExposureInuAs
    except:
        count_exceptions2 += 1
    try:
        d["RelativeXRayExposure"] = A.RelativeXRayExposure
    except:
        count_exceptions3 += 1

    return d


def read_IMG_from_folder(DATA_DIR, folder):

    # Read IMG in given path and folder

    FOLDER = os.path.join(DATA_DIR, folder)
    img = os.listdir(FOLDER)[0]
    IMG_DIR = os.path.join(FOLDER, img)
    IMG = pydicom.dcmread(IMG_DIR)
    return IMG


def extract_id_from_image(img):

    # Returns the patient id of img.

    r = re.compile("^ST_([^_]*)_.*")
    var = r.match(img)
    if var is not None:
        return int(var.groups(1)[0])
    return -1


def valid_image(img, lista):

    # Checks if the img patient id is in lista.

    id_paciente = extract_id_from_image(img)
    return id_paciente in lista


def invert(img, orientation):

    # If orientation is MONOCHROME1 img colors are inverted and its values are transformed to nummbers between 0 and 255.

    if orientation == "MONOCHROME1":
        img = cv2.bitwise_not(img)
        img = img - np.amin(img)
        img = (img / np.amax(img)) * 255
    return img


def adjust(img):
    """Img values are transformed to nummbers between 0 and 255."""

    img = img - np.amin(img)
    img = (img / np.amax(img)) * 255
    return img


def image_to_test(img):

    # Prepares img for the neural network.

    npy = img / 255
    npy = np.reshape(npy, npy.shape + (1,))
    npy = np.reshape(npy, (1,) + npy.shape)
    return npy


def segment(im):
    """Labels pixels of im by colors into connected groups."""

    groups = im > 0.6 * im.mean()
    return measure.label(groups)


def chose_segments(im):

    # with im given, returns another with the two biggest groups by color,
    # changing the rest of pixels to 0. Its also returned the number of clusters
    # in the original im and a list with the number of pixels in each one of the big clusters.
    # If the second group were to small in comparation with the first, its transformed to 0
    # (the groups we look for are connected in one).

    segmentos = segment(im)
    final_mask = im
    l = np.unique(segmentos)
    g = []
    for i in np.unique(segmentos):
        g.append(np.count_nonzero(segmentos == i))

    areas = []
    s = []
    for z in range(min(3, len(g))):
        if z == 1:
            areas.append(np.amax(g) / (512 ** 2))
            s.append(l[np.where(g == np.amax(g))])
        elif z == 2:
            if (np.amax(g) / (512 ** 2)) / areas[0] >= 0.25:
                areas.append(np.amax(g) / (512 ** 2))
                s.append(l[np.where(g == np.amax(g))])
        l = np.delete(l, np.where(g == np.amax(g)))
        g = np.delete(g, np.where(g == np.amax(g)))

    for a in range(512):
        for b in range(512):
            if segmentos[a, b] in s:
                final_mask[a, b] = im[a, b]
            else:
                final_mask[a, b] = 0
    num_segments = len(areas)

    return (final_mask, num_segments, areas)


def pmedia(final_mask, img, lung_area, mode="probabilidad"):
    """Calculates the mean of img in final_mask. If mode = 'probabilidad' final_mask is used as weights."""

    if lung_area != 0:
        if mode == "probabilidad":
            mat = np.multiply(final_mask, img)
            suma = np.sum(mat)
        else:
            final_mask = np.where(final_mask > 0.5, 1, 0)
            important_pixels = np.multiply(final_mask, img)
            suma = np.sum(important_pixels)

        lung_mean = suma / lung_area
    else:
        lung_mean = 0

    return lung_mean


def brighter(final_mask, img, x, mode="probabilidad"):

    # Number of img pixels inside final_mask with greater value than x. If mode = 'probabilidad' final_mask is used as weights.

    if mode == "probabilidad":
        mat = np.where(img > x, final_mask, 0)
        ret = np.sum(mat)
    else:
        important_pixels = np.multiply(final_mask, img)
        ret = np.count_nonzero(important_pixels > x)

    return ret


def mean_brighter(final_mask, img, x, num_pixels, mode="probabilidad"):

    # Mean of img pixels in final_mask with greater value than x. If mode = 'probabilidad' final_mask is used as weights.

    if num_pixels != 0:
        if mode == "probabilidad":
            mat = np.where(img > x, img, 0)
            mat = np.multiply(mat, final_mask)
            suma = np.sum(mat)
        else:
            important_pixels = np.multiply(final_mask, img)
            important_pixels = np.where(important_pixels <= x, 0, important_pixels)
            suma = np.sum(important_pixels)
        mean = suma / num_pixels
    else:
        mean = 0

    return mean


def envoltura_convexa(img):

    # Convex wrap of each cluster in img.

    segmentos = segment(img)
    l = np.unique(segmentos)
    g = np.zeros(img.shape)
    mat = np.zeros(img.shape)
    for i in range(len(l) - 1):
        mat = np.where(segmentos == l[i + 1], 1, 0)
        f = convex_hull_image(mat)
        g = np.where(f, 1, g)
    return g


def find_transformation(default_lungs, segm_ret):

    # Search for the best afin transformation which applied to default_lungs gets segm_ret. Returns default_lungs with the transformation found.

    a = 1
    image_size = segm_ret.shape
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(
            segm_ret, default_lungs, warp_matrix, cv2.MOTION_AFFINE, criteria
        )
        default_lungs = cv2.warpAffine(
            default_lungs,
            warp_matrix,
            (image_size[1], image_size[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except:
        default_lungs = np.zeros((image_size[1], image_size[0]))
        a = 2

    return [default_lungs, a, warp_matrix]


def get_real_mask(default_lungs, segm_ret):

    # Weighted combination of segm_ret, its convex wrap and the best afin transformation of default_lungs (to get segm_ret).

    b = 1
    [final_mask_1, a, warp_matrix] = find_transformation(default_lungs, segm_ret)
    [final_mask_2, c, _] = chose_segments(segm_ret)
    if c == 1:
        b = 0
        final_mask_3 = np.zeros((final_mask_2.shape[1], final_mask_2.shape[0]))
    else:
        final_mask_3 = envoltura_convexa(segm_ret)
        final_mask_3 = chose_segments(final_mask_3)[0]

    final_mask = (final_mask_1 * a * 2 + final_mask_2 + final_mask_3 * b) / (
        1 + 2 * a + b
    )

    return final_mask, warp_matrix


def add_colored_mask(image, mask):
    """Combines image (input in color) and mask (input in gray scale)."""
    X = np.stack([mask, mask * 0, mask * 0], axis=2)
    ret = (image * 0.7 + X * 0.3) / 255
    return ret


def plot_both(mask, img):
    """Given mask and img, both in gray scale, are combined changing mask to red. The combinated image is shown."""

    mask = np.uint8(mask)
    image = img
    image = np.uint8(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ret = add_colored_mask(image, mask)
    plt.imshow(ret)
    plt.show()


def normalize(img):
    """Values of img are transformed to numbers between 0 and 1, top 0.5% and bottom 0.5% are truncated."""

    a = np.percentile(img, 0.5)
    b = np.percentile(img, 99.5)
    img = np.where(img > b, b, img)
    img = np.where(img < a, a, img)
    img = (img - a) / (b - a)

    return img


def fmedia(img):
    """Brigthness mean of img (without 0)."""

    count = np.count_nonzero(img > 0)
    if count != 0:
        suma = np.sum(img)
        ret = suma / count
    else:
        ret = 0

    return ret


def meanbrt(img, brt, x):
    """Brigthness mean of img pixels with greater value than x, brt is the area of those pixels"""

    mat = np.where(img > x, 1, 0)
    mat = np.multiply(img, mat)
    suma = np.sum(mat)

    return suma / brt


def area_histograma(img, final_mask):

    # Ratio of histogram area of img and img weighted by final_mask that match.

    image = img.flatten()
    mask = final_mask.flatten()[np.where(image > 0)]
    image = image[np.where(image > 0)]
    uno = np.histogram(image, density=True, bins=200)
    dos = np.histogram(image, weights=mask, density=True, bins=200)
    suma = 0
    suma_dos = 0
    for i in range(len(uno[0])):
        suma += min(uno[0][i], dos[0][i])
        suma_dos += dos[0][i]
    return suma / suma_dos


def separar(mask):
    """Returns each lung in mask in a different mask."""

    segmentos = segment(mask)
    mask1 = np.where(segmentos == 1, mask, 0)
    mask2 = np.where(segmentos == 2, mask, 0)
    return mask1, mask2


def median(mask, img):

    """Calculates the median of img in mask."""

    hel = np.where(mask > 0.25, img, np.nan)
    hel = np.nanmedian(hel)
    return hel


def SliceMask(mask):

    # Separates mask in three parts of equal height.

    indices = np.indices(mask.shape)[0]
    p, g = np.min(np.where(mask > 0.1)[0]), np.max(np.where(mask > 0.1)[0])
    supmask = np.where((indices > p) & (indices < ((g - p) / 3 + p)), mask, 0)
    midmask = np.where(
        (indices > ((g - p) / 3 + p)) & (indices < (2 * (g - p) / 3 + p)), mask, 0
    )
    submask = np.where((indices > (2 * (g - p) / 3 + p)) & (indices < g), mask, 0)

    return supmask, midmask, submask
