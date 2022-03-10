import cv2
from importlib_metadata import files
import matplotlib.pyplot as plt
import os
import random
import matplotlib.pyplot as plt
import config
import cv2
import math
import mediapipe as mp
import numpy as np
import json
import scipy.spatial.distance as distance
from scipy.spatial import ConvexHull
import random


def get_segmentations(id) -> dict:
    """
    Returns:
        dict: hand_id: str  -> list_of_contours (each contour is a list of points): list[list[tuple(float, float)]]
    """
    with open(f"data/jsons/{id}.json", "r") as f:
        file = json.load(f)
    first_key = list(file.keys())[0]
    # bone_ids = [hand_info["id"] for hand_info in file["items"]]
    regions = file[first_key]["regions"]
    segmentations = {
        region_idx: [
            (
                regions[region_idx]["shape_attributes"]["all_points_x"][point_idx],
                regions[region_idx]["shape_attributes"]["all_points_y"][point_idx],
            )
            for point_idx in range(
                len(regions[region_idx]["shape_attributes"]["all_points_x"])
            )
        ]
        for region_idx in range(len(regions))
    }
    return segmentations


def _draw_point(img, point):
    return cv2.circle(
        img,
        (int(point[0]), int(point[1])),
        radius=0,
        color=(0, 255, 0),
        thickness=3,
    )


def _draw_contour(img, list_of_points, bone_num=""):
    center_x = int(np.mean([p[0] for p in list_of_points]))
    center_y = int(np.mean([p[1] for p in list_of_points]))
    fontScale = 3
    fontFace = cv2.FONT_HERSHEY_PLAIN
    fontColor = (0, 255, 255)
    fontThickness = 2
    cv2.putText(
        img,
        bone_num,
        (center_x, center_y),
        fontFace,
        fontScale,
        fontColor,
        fontThickness,
        cv2.LINE_AA,
    )
    for point in list_of_points:
        img = _draw_point(img, point)
    return img


def draw_all_contours(img, segmentations, order=None):
    segmentations = (
        segmentations
        if order is None
        else {index: segmentations[id] for index, id in enumerate(order)}
    )
    for idx, contour in segmentations.items():
        print(idx)
        img = _draw_contour(img, contour, str(idx))
    return img


def _get_convex_hull(list_of_points):
    cvx_hull = cv2.convexHull(np.array(list_of_points).astype(np.float32))
    cvx_hull = [(p[0][0], p[0][1]) for p in cvx_hull]
    return cvx_hull


def get_diameter(list_of_points):
    pairwise_distances = distance.pdist(np.array(list_of_points))
    return np.max(pairwise_distances)


def get_distance_between_contours(contour1, contour2):
    pairwise_distances = distance.cdist(np.array(contour1), np.array(contour2))
    return np.min(pairwise_distances)


def get_perimeter(list_of_points):
    ch = ConvexHull(np.array(list_of_points).astype(np.float32))
    # NOTE: ConvexHull().area is actually the perimeter if the input
    # matrix is 2dimensional
    return ch.area


BGR_color_bounds = {
    "yellow": {"lower": (0, 170, 170), "upper": (50, 255, 255)},
    "green": {"lower": (115, 185, 115), "upper": (170, 255, 185)},
    "red": {"lower": (0, 0, 180), "upper": (100, 100, 255)},
    "purple": {"lower": (140, 0, 140), "upper": (210, 75, 205)},
    "cyan": {"lower": (200, 185, 0), "upper": (255, 255, 65)},
}


def has_color(colored_img, segment, color):
    # print(colored_img)
    # print(colored_img.shape)
    # print(segment)
    # print(color)

    color_bounds = BGR_color_bounds[color]
    # !!! WARNING: the first component of an array is the y-axis component!!!
    segment_pixels = [colored_img[p[1], p[0], :] for p in segment]
    return np.mean(
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
    )> 0.25  # If we don't put a large enough lower bound, then this will detect off-color bones...# that slighlty touch on-color bounds


def apply_warp(img_gray, matrix):
    return cv2.warpAffine(
        img_gray,
        matrix,
        img_gray.shape,
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def detect_color_segments(id, segments, color):
    color_segments = {}
    colored_img = cv2.imread(os.path.join(config.colored_data_dir, f"{id}.png"))
    if config.do_affine_transform:
        matrix = np.load(os.path.join(config.affine_matrices_dir, f"{id}_matrix.npy"))
        b, g, r = cv2.split(colored_img)
        b, g, r = apply_warp(b, matrix), apply_warp(g, matrix), apply_warp(r, matrix)
        colored_img = cv2.merge((b, g, r))
    for segment_id, segment in segments.items():
        if has_color(colored_img, segment, color):
            color_segments[segment_id] = segment
    return color_segments


if __name__ == "__main__":

    def inspect_json(id):

        with open(f"data/jsons/{id}.json", "r") as f:
            file = json.load(f)
        first_key = list(file.keys())[0]
        id = first_key.split(".")[0]
        return id
        img = cv2.imread(f"data/boneage-training-dataset/{id}.png", 1)
        segmentations = get_segmentations(id)

        for idx, contour in segmentations.items():
            img = _draw_contour(img, contour, str(idx))

        for idx, contour in segmentations.items():
            diameter = get_diameter(contour)
            # print(f"diameter of {idx}: {diameter}")

        # plt.imshow(img)
        # plt.title(id)
        # plt.show()

    files_list = os.listdir("data/jsons")
    keys = []
    weird_cases = []
    random.shuffle(files_list)
    for file_name in files_list:
        key = inspect_json(file_name.split(".")[0])
        keys.append(key)
        if key.split(".")[0] != file_name.split(".")[0]:
            weird_cases.append(key)
            os.remove(os.path.join("data/jsons", file_name))
    print(len(files_list))
    print(len(weird_cases))
    print(len(set(keys)))
