import matplotlib.pyplot as plt
import config
import cv2
import math
import mediapipe as mp
from utils import annotate_img
import numpy as np


def no_landmarks_wrapper(fun):
    """A function wrapped with this will skip itself and
    return None if `self.landmarks` is None"""

    def wrapped_fun(*args, **kwargs):
        if args[0].landmarks is None:
            return None
        else:
            return fun(*args, **kwargs)

    return wrapped_fun


class dotdict(dict):
    """Any class inheriting from this will be a dictionary whose attributes can be accessed with .dot notation"""

    # TODO: Used?
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


mp_hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class HandInterface(object):
    # TODO: how do interfaces work really?
    def __init__(self):
        for feature in config.ALL_FEATURE_NAMES:
            setattr(self, feature, None)


class Hand(HandInterface):
    def __init__(self, _image, _boneage, _gender, _id):

        self.img = _image
        self.boneage = _boneage
        self.gender = _gender
        self.id = _id
        self.raw_landmarks = None  # landmarks in the format given by mediapipe
        self.landmarks = None  # landmarks in a dictionary form more comfortable for us

        # Attributes used internally
        self._middle_finger_length = None
        self._gap_proxy_mean = None
        self._gap_proxy_std = None

        # Populate attributes
        self.get_hand_landmarks()
        self.get_gap_bones_proxy()
        self.featurize()

    def get_hand_landmarks(self):
        # TODO: refactor this
        """Returns a dictionary with items of the form:
        landmark_id (int) : (x_coordinate, y_coordinate)
        """
        dict_centered_landmarks = {}
        img = self.img
        mp_result = mp_hands.process(img)
        # mp_result.multi_hand_landmarks is None if no landmarks were found
        # Otherwise it is a list (?) with length the number of hands detected
        # each entry is a list of a particular hand landmark
        if mp_result.multi_hand_landmarks is None:
            print(
                f"No hand landmarks were found in Hand {self.id} "
                f"with boneage {self.age} and gender {self.gender}"
            )
            return None
        # We store the hand_landmark data structure given my mediapipe as `raw_landmarks`
        # This is only used so far in the `draw_landmarks` method
        # We will be working with the dictionary version that we create now
        # The zero is because there will always be only one hand
        self.raw_landmarks = mp_result.multi_hand_landmarks[0]
        landmarks = mp_result.multi_hand_landmarks[0].landmark
        for id, landmark in enumerate(landmarks):
            # Here we scale the point to match the img size
            height, width, channels = self.img.shape
            x_img = int(landmark.x * width)
            y_img = int(landmark.y * height)
            # Now we add the point to the dictionary
            dict_centered_landmarks[id] = (x_img, y_img)
            if config.annotate_imgs:
                # And we write the landmark id on the image
                annotate_img(
                    self.img,
                    (x_img, y_img),
                    str(id),
                )
        self.landmarks = dict_centered_landmarks

    def featurize(self):
        for feature_name in config.ALL_FEATURE_NAMES:
            feature_value = eval(f"self.get_{feature_name}()")
            setattr(self, feature_name, feature_value)

    def get_boneage(self):
        return self.boneage

    def get_gender(self):
        return self.gender

    @no_landmarks_wrapper
    def get_ratio_finger_palm(self):
        dict_landmarks = self.landmarks
        # tip of mifflr finger to palm
        middle_finger_landmark_ids = [12, 11, 10, 9]
        # left to right of the palm
        top_palm_landmark_ids = [17, 13, 9, 5]
        middle_finger_length = get_consecutive_ldk_distances(
            dict_landmarks, middle_finger_landmark_ids
        )
        top_palm_length = get_consecutive_ldk_distances(
            dict_landmarks, top_palm_landmark_ids
        )
        ratio = middle_finger_length / top_palm_length
        self._middle_finger_length = middle_finger_length
        if config.annotate_imgs:
            annotate_img(self.img, (0, 1600), f"finger_to_palm {round(ratio,2)}")
        return ratio

    @no_landmarks_wrapper
    def get_gap_bones_proxy(self):
        ldk_10 = self.landmarks[10]
        distance_10_9 = get_distance(self.landmarks[10], self.landmarks[9])
        ratio = 0.17
        _img_copy = self.img.copy()
        displacement = math.floor(ratio * distance_10_9)
        # NOTE: the fist dimension is the height
        square = _img_copy[
            ldk_10[1] - displacement : ldk_10[1] + displacement,
            ldk_10[0] - displacement : ldk_10[0] + displacement,
        ]
        square = cv2.cvtColor(square, cv2.COLOR_RGB2GRAY)
        square = cv2.equalizeHist(square)
        square = cv2.cvtColor(square, cv2.COLOR_GRAY2BGR)

        self._gap_proxy_mean = square.mean()
        self._gap_proxy_std = square.std()

        if config.annotate_imgs:
            point_low_left = (ldk_10[0] - displacement, ldk_10[1] - displacement)
            point_up_right = (ldk_10[0] + displacement, ldk_10[1] + displacement)
            cv2.rectangle(
                self.img,
                point_low_left,
                point_up_right,
                (255, 0, 255),
                thickness=2,
                lineType=cv2.LINE_8,
            )

    @no_landmarks_wrapper
    def get_ratio_finger_to_gap_std(self):
        return self._middle_finger_length / self._gap_proxy_std

    @no_landmarks_wrapper
    def get_ratio_finger_to_gap_mean(self):
        return self._middle_finger_length / self._gap_proxy_mean

    """----------------------------------------------------------------
    Utility methods 
    ----------------------------------------------------------------"""

    def draw_landmarks(self, _hand):
        if _hand.raw_landmarks is None:
            print("No hand landmarks detected")
            return None
        mp_draw.draw_landmarks(
            _hand.img,
            _hand.raw_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    def show(self):
        plt.imshow(self.img)
        plt.title(f"Hand id {self.id}, boneage {self.age}, gender {self.gender}")
        plt.show()


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
