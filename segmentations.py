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
    segmentations = segmentations if  order is None else {index: segmentations[id] for index, id in enumerate(order)}
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


if __name__ == "__main__":

    def inspect_json(id):
        
        with open(f"data/jsons/{id}.json", "r") as f:
            file = json.load(f)
        first_key = list(file.keys())[0]
        id = first_key.split('.')[0]
        img = cv2.imread(f"data/boneage-training-dataset/{id}.png", 1)
        segmentations = get_segmentations(id)

        for idx, contour in segmentations.items():
            img = _draw_contour(img, contour, str(idx))

        for idx, contour in segmentations.items():
            diameter = get_diameter(contour)
            print(f"diameter of {idx}: {diameter}")


        plt.imshow(img)
        plt.title(id)
        plt.show()

    files_list = os.listdir("data/jsons")
    keys = []
    weird_cases = []
    random.shuffle(files_list)
    for file_name in files_list:
        key = inspect_json(file_name.split(".")[0])
        
        
    
    #     keys.append(key)
    #     if key.split(".")[0] != file_name.split('.')[0]: 
    #         weird_cases.append(key)
    # 
    # print(len(files_list))
    # print(len(weird_cases))
    # print(len(set(keys)))