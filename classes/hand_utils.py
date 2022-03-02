from tkinter import E
from scipy.spatial import procrustes
import numpy as np
import cv2
import matplotlib.pyplot as plt

def no_landmarks_wrapper(fun):
    """A function wrapped with this will skip itself and
    return None if `self.landmarks` is None"""

    def wrapped_fun(*args, **kwargs):
        if args[0].landmarks is None:
            return None
        else:
            return fun(*args, **kwargs)

    return wrapped_fun

def get_distance(point1, point2):
    # point = tuple(int, int)
    translated_point = (point1[0] - point2[0], point1[1] - point2[1])
    return np.sqrt(translated_point[0] ** 2 + translated_point[1] ** 2)


def get_consecutive_ldk_distances(_dict_landmarks, _landmark_ids_list):
    total_distance = 0
    for idx, ldk_id in enumerate(_landmark_ids_list[:-1]):
        next_ldk_id = _landmark_ids_list[idx + 1]
        distance = get_distance(_dict_landmarks[ldk_id], _dict_landmarks[next_ldk_id])
        # print(distance, ldk_id, next_ldk_id)
        total_distance += distance
    return total_distance


def align_batch_of_hands(_hand_list):
    array_2_2 = prepare_for_procusters(_hand_list)
    print(array_2_2.shape)
    mean_img = array_2_2.mean(axis=0)
    print(mean_img.shape)
    mean_img_2_2 = np.repeat(mean_img[np.newaxis, :], array_2_2.shape[0], axis=0)
    print(mean_img_2_2.shape)
    print(mean_img_2_2)
    
    mtx1, mtx2, disp = procrustes(array_2_2,mean_img_2_2)
    print(mtx1)
    print(mtx2)
    transformed_hand_list = undo_procusters_preparation(mtx1, mtx2)
    return transformed_hand_list



resize_factor = 0.025


def prepare_for_procusters(_hand_list):
    img_arrays = [h.img for h in _hand_list]
    
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


def undo_procusters_preparation(mtx1, mtx2):
    print(mtx2)
    for x in mtx2:
        print(x)
        x = x.reshape(int(1668*0.025), int(0.025*2066))
        plt.imshow(x)
        plt.show()
    
    
