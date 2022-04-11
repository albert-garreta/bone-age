from gc import collect
import cv2
import numpy as np
import json
import scipy.spatial.distance as distance
from scipy.spatial import ConvexHull


"""
This file contains functions for working with the contours of bones provided 
in the json files (these files have been obtained privately and for the moment are not publicly available).
Each json file has the name <HAND_ID>.json. Such a file contains the bone contours of the hand given in
<HAND_ID>.png. A contour is a list of points.

Here we provide methods for: 
    - Extracting the contours from the json files,
    - Finding the diameter and the perimeter of a contour
    - Finding the distance between two contours
    - Detecting the type of contour: for each <HAND_ID>.json we also
    have a private .png file where bones have been colored with distinct colors
    depending on the type of bone (there are 5 types of bones in total). 
    
    See the `has_color` function.
"""


def get_segmentations(id, hand_dimensions=None) -> dict:
    """
    Given a hand id (e.g. 1712), gets all the bone contours provided in the corresponding json file
    (following the example, this would be 1712.json)
    for the hand. A contour is a list of points surrounding a bone. For each hand, this function
    returns a dictionary mapping the order of appearance of a contour in the json file, to the
    contour.

    NOTE: This function accepts (optionally) the dimensions of image <ID>.png as input
    because sometimes the contours have some coordinates which are off the image by one pixel,
    and thus they need to be clipped. If `hand_dimensions` is not provided then the function
    ignores this, but then one may get a `list index out of range` error later on.

    Returns:
        dict: hand_id: str  -> list_of_contours (each contour is a list of points): list[list[tuple(float, float)]]
    """
    with open(f"data/jsons/{id}.json", "r") as f:
        file = json.load(f)

    # From here on it's just about extracting the contours from the json in the specified dict format
    first_key = list(file.keys())[0]
    regions = file[first_key]["regions"]
    if hand_dimensions is not None:
        hand_x_dim, hand_y_dim = hand_dimensions[:2]
    else:
        hand_x_dim, hand_y_dim = np.inf, np.inf
    segmentations = {
        region_idx: [
            (
                # WARNING: sometimes the points are out of the image by one pixel.
                # For this reason we make this max-min clipping
                np.clip(
                    regions[region_idx]["shape_attributes"]["all_points_x"][point_idx],
                    0,
                    hand_y_dim,
                ),
                np.clip(
                    regions[region_idx]["shape_attributes"]["all_points_y"][point_idx],
                    0,
                    hand_x_dim,
                ),
            )
            for point_idx in range(
                len(regions[region_idx]["shape_attributes"]["all_points_x"])
            )
        ]
        for region_idx in range(len(regions))
    }
    return segmentations


def get_diameter(list_of_points):
    # Gets the diameter of a list of points
    pairwise_distances = distance.pdist(np.array(list_of_points))
    return np.max(pairwise_distances)


def get_distance_between_two_lists_of_points(list1, list2):
    pairwise_distances = distance.cdist(np.array(list1), np.array(list2))
    return np.min(pairwise_distances)


def get_perimeter(list_of_points):
    ch = ConvexHull(np.array(list_of_points).astype(np.float32))
    # NOTE: ConvexHull().area is actually the perimeter if the input
    # matrix is 2dimensional, as it is in our case
    return ch.area


def draw_all_contours(img, segmentations, color=None, write_contour_number=False):
    for idx, contour in segmentations.items():
        write_contour_number = str(idx) if write_contour_number else None
        img = _draw_contour(img, contour, write_contour_number, color)
    return img


def _draw_contour(img, list_of_points, text=None, color=None):
    """Given an image and a list of points, draws each point on the
    image. Optionally it writes a piece of text on the mean point of
    the set of points"""
    if text is not None:
        center_x = int(np.mean([p[0] for p in list_of_points]))
        center_y = int(np.mean([p[1] for p in list_of_points]))
        fontScale = 3
        fontFace = cv2.FONT_HERSHEY_PLAIN
        fontColor = (0, 255, 255)
        fontThickness = 2
        cv2.putText(
            img,
            text,
            (center_x, center_y),
            fontFace,
            fontScale,
            fontColor,
            fontThickness,
            cv2.LINE_AA,
        )
    for point in list_of_points:
        img = _draw_point(img, point, color)
    return img


def _draw_point(img, point, color):
    return cv2.circle(
        img,
        (int(point[0]), int(point[1])),
        radius=0,
        color=color,
        thickness=3,
    )


# Next we write a function that allows to detect the color of a contour according to the
# first batch of "tagged" hands we got (see the folder `tagged_data_colored_contours`).
# We start by specifying BGR bounds for each color.
BGR_color_bounds = {
    "yellow": {"lower": (0, 170, 170), "upper": (50, 255, 255)},
    "green": {"lower": (115, 185, 115), "upper": (170, 255, 185)},
    "red": {"lower": (0, 0, 120), "upper": (120, 120, 255)},
    "purple": {"lower": (120, 0, 120), "upper": (255, 120, 255)},
    "cyan": {"lower": (200, 185, 0), "upper": (255, 255, 65)},
}


def has_color(colored_img, contour, color):
    """Given an image `colored_img` with colored bone contours (as in the images from the folder
    `tagged_data_colored_contours`, and a contour (i.e. a list of points), and a color (e.g. "yellow")
    the function determines wether or not the contour is of the specified color in `colored_img`

    TODO: I think that the current method for detecting the colors appearing in a contour
    can be improved significantly. However, it is not completely trivial (i.e. the pixels to detect
    don't always have a "pure" color --sometimes a pixel has a dimmed green instead of a clear green, etc---
    I suspect this may be caused because the points from the json files are sometimes offset by one (or more?)
    pixels --see the note about clipping in the function `get_segmentations`--- though I'm not completely sure)
    """

    color_bounds = BGR_color_bounds[color]
    
    # Next we get a list of pairs of pixels corresponding to the points in the contour
    # from the json file.

    # See the remark about clipping in the function `get_segmentations`
    #clipped_contour = [(np.clip(p[0], 0, colored_img.shape[0]), np.clip(p[1], 0, colored_img.shape[1])) for p in contour]
    clipped_contour = np.clip(np.array(contour), 0, list(colored_img.shape[:2]))
    
    # !!! WARNING: the first component of an array is the y-axis component in the Euclidean plane
    segment_pixels = [colored_img[p[1], p[0], :] for p in clipped_contour]
    return (
        np.mean(
            [
                all(
                    [
                        pixel[channel] > color_bounds["lower"][channel]
                        for channel in range(3)
                    ]
                    + [
                        pixel[channel] < color_bounds["upper"][channel]
                        for channel in range(3)
                    ]
                )
                for pixel in segment_pixels
            ]
        )
        > 0.25
    )  # If we don't put a large enough lower bound, then this will detect off-color bones...# that slighlty touch on-color bounds
