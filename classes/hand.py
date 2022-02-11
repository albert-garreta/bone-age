import matplotlib.pyplot as plt
import config
import cv2
import math
import mediapipe as mp
from scripts.utils import annotate_img
import numpy as np
from classes.hand_utils import (
    no_landmarks_wrapper,
    get_distance,
    get_consecutive_ldk_distances,
)

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
        # mediapipe hand landmarks in the format given by mediapipe
        self.raw_landmarks = None
        # mediapipe hand landmarks in a dictionary form more comfortable for us
        self.landmarks = None

        # Attributes used internally
        self.length_middle_finger = None
        self._gap_proxy_mean = None
        self._gap_proxy_std = None

        self._populate_attributes()

    def _populate_attributes(self):
        # TODO: clean this _ naming stuff
        self._get_hand_landmarks()
        self.get_gap_bones_proxy()
        self.featurize()

    """----------------------------------------------------------------
    Get mediapipe's hand lanmarks methods
    ----------------------------------------------------------------"""

    def _get_hand_landmarks(self):
        """Creates the attribute `landmarks`: a dictionary with items of the form:
        landmark_id (int) : (x_coordinate, y_coordinate)
        """
        raw_landmarks, success = self._get_raw_landmarks()
        if not success:
            print(
                f"No hand landmarks were found in Hand {self.id} "
                f"with boneage {self.boneage} and gender {self.gender}"
            )
        self.raw_landmarks = raw_landmarks
        self._convert_raw_landmarks()

    def _get_raw_landmarks(self):
        mp_result = mp_hands.process(self.img)
        if mp_result.multi_hand_landmarks is None:
            return None, False
        else:
            return mp_result.multi_hand_landmarks[0], True

    def _convert_raw_landmarks(self):
        landmarks = self.raw_landmarks.landmark
        self.landmarks = {}
        for id, landmark in enumerate(landmarks):
            self.process_individual_landmark(id, landmark)

    def process_individual_landmark(self, _id, _landmark):
        x_scaled, y_scaled = self.get_scaled_landmark_coordinates(_landmark)
        self.landmarks[_id] = (x_scaled, y_scaled)
        if config.annotate_imgs:
            # Write the landmark id on the image
            annotate_img(
                self.img,
                (x_scaled, y_scaled),
                str(_id),
            )

    def get_scaled_landmark_coordinates(self, _landmark):
        height, width, channels = self.img.shape
        x_scaled = int(_landmark.x * width)
        y_scaled = int(_landmark.y * height)
        return x_scaled, y_scaled

    """----------------------------------------------------------------
    Feature creation methods
    ----------------------------------------------------------------"""

    def featurize(self):
        """Main function to create the features.
        The features used are customizable from the config file.
        For this reason the class uses the keywords `eval`, `setattr`, and
        `getattr` which allow to evaluate functions, set attributes, or get
        attributes by passing in the string representation of the function
        or attributes being used.
        """
        for feature_name in config.ALL_FEATURE_NAMES:
            feature_value = eval(f"self.get_{feature_name}()")
            setattr(self, feature_name, feature_value)

    def get_boneage(self):
        # These are redundant but we need them for our customizable features logic
        return self.boneage

    def get_gender(self):
        # These are redundant but we need them for our customizable features logic
        return self.gender

    @no_landmarks_wrapper
    def get_length_middle_finger(self):
        middle_finger_landmark_ids = [12, 11, 10, 9]
        self.length_middle_finger = get_consecutive_ldk_distances(
            self.landmarks, middle_finger_landmark_ids
        )

    @no_landmarks_wrapper
    def get_length_top_palm(self):
        top_palm_landmark_ids = [17, 13, 9, 5]
        self.length_top_palm = get_consecutive_ldk_distances(
            self.landmarks, top_palm_landmark_ids
        )
    
    @no_landmarks_wrapper
    def get_ratio_finger_palm(self):
        return self.length_middle_finger / self.length_top_palm

    @no_landmarks_wrapper
    def get_gap_bones_proxy(self):
        # TODO: clean this up
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

    def get_ratio_finger_to_gap_std(self):
        return self.length_middle_finger / self._gap_proxy_std

    def get_ratio_finger_to_gap_mean(self):
        return self.length_middle_finger / self._gap_proxy_mean

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
