import os

from regex import X
import config
import cv2
import numpy as np

def remove_file_types_from_list(list_of_files):
    return [f.split(".")[0] for f in list_of_files]

def get_list_hand_files():
    # Gets all the hand files (in the form <HAND_ID.png> for which we have a contour)
    
    # We restrict to hand ids for which we have a json with the annotated bone contours
    list_json_files = os.listdir(config.jsons_folder)
    list_json_files = [
        file for file in list_json_files if file != ".DS_Store"
    ]
    list_json_files.sort()
    list_ids = remove_file_types_from_list(list_json_files)
    list_hand_files = [f"{id}.png" for id in list_ids]
    return list_hand_files


def get_valid_hand_ids():
    # Takes the list from the previous function and removes the hands with id
    # belonging to `config.FORBIDDEN_IMGS`
    list_hand_files = get_list_hand_files()
    list_ids = remove_file_types_from_list(list_hand_files)
    valid_ids = [id for id in list_ids if id not in config.FORBIDDEN_IMGS]
    return valid_ids


def annotate_img(img, point, text):
    cv2.putText(
        img,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 255),
        1,
    )

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)


COLOR_NAME_TO_BGR = {
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "purple": (255, 0, 255),
    "yellow": (0, 255, 255),
}
