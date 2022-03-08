import cv2
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

def get_segmentations(id) -> dict:
    """
    Returns:
        dict: hand_id: str  -> list_of_contours (each contour is a list of points): list[list[tuple(float, float)]]
    """
    with open(f"{id}.json", "r") as f:
        file = json.load(f)
    bone_ids = [hand_info["id"] for hand_info in file["items"]]
    segmentations = {
        bone_ids[idx]: [
            [
                (info["points"][idx], info["points"][idx + 1])
                for idx in range(0, len(info["points"]), 2)
            ]
            for info in file["items"][idx]["annotations"]
        ]
        for idx in range(len(bone_ids))
    }
    return segmentations


def draw_point(img, point):
    return cv2.circle(
        img,
        (int(point[0]), int(point[1])),
        radius=0,
        color=(0, 255, 0),
        thickness=3,
    )


def draw_contour(img, list_of_points, bone_num=""):
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
        img = draw_point(img, point)
    return img


def get_convex_hull(list_of_points):
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
    id = 10214
    img = cv2.imread(f"data/boneage-training-dataset/{id}.png", 1)
    segmentations = get_segmentations(id)

    segmentations_id = segmentations[str(id)]

    for idx, contour in enumerate(segmentations_id):
        img = draw_contour(img, contour, str(idx))

    for idx, contour in enumerate(segmentations_id):
        diameter = get_diameter(contour)
        print(f"diameter of {idx}: {diameter}")

    print(
        "Distance between 0st and 9th bone: ",
        get_distance_between_contours(segmentations_id[0], segmentations_id[9]),
    )

    plt.imshow(img)
    plt.show()
