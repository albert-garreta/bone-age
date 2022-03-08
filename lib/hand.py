import matplotlib.pyplot as plt
import scripts.config as config
import cv2
import math
import mediapipe as mp
from utils import annotate_img, get_line_function
import numpy as np
from classes.hand_utils import (
    no_landmarks_wrapper,
    get_distance,
    get_consecutive_ldk_distances,
)

mp_hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class Hand(object):
    def __init__(self, _image, _boneage, _gender, _id, _segmented_img):

        self.img = _image
        self.segmented_img = _segmented_img
        self.boneage = _boneage
        self.gender = _gender
        self.id = _id
        # mediapipe hand landmarks in the format given by mediapipe
        self.raw_landmarks = None
        # mediapipe hand landmarks in a dictionary form more comfortable for us
        self.landmarks = None

        # Attributes used internally
        self._length_middle_finger = None
        self._gap_proxy_mean = None
        self._gap_proxy_std = None
    
    
    """----------------------------------------------------------------
    Feature creation methods
    ----------------------------------------------------------------"""

    def featurize(self) -> bool:
        """Main function to create the features.
        The features used are customizable via string types from the config file.
        
        ** To set an attribute, in the config file add its `<name>` in the list
        `ALL_FEATURE_NAMES`. Then implement a method here with name `get_<name>` **
        
        For this reason the class uses the keywords `eval`, `setattr`, and
        `getattr` which allow to evaluate functions, set attributes, or get
        attributes by passing in the string representation of the function
        or attributes being used.
        """
        for feature_name in config.ALL_FEATURE_NAMES:
            try:
                # print(feature_name)
                feature_value = eval(f"self.get_{feature_name}()")
                setattr(self, feature_name, feature_value)
            except Exception as e:
                print(e)
                return False
        return True

    def get_boneage(self):
        return self.boneage

    def get_gender(self):
        return self.gender

    
    
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
            return None
        else:
            self.raw_landmarks = raw_landmarks
            self._convert_raw_landmarks()

    def _get_raw_landmarks(self):
        mp_result = mp_hands.process(self.img)
        if mp_result.multi_hand_landmarks is None:
            return None, False
        else:
            return mp_result.multi_hand_landmarks[0], True

    # @no_landmarks_wrapper
    def _convert_raw_landmarks(self):
        landmarks = self.raw_landmarks.landmark
        self.landmarks = {}
        for id, landmark in enumerate(landmarks):
            self._process_individual_landmark(id, landmark)

    def _process_individual_landmark(self, _id, _landmark):
        x_scaled, y_scaled = self._get_scaled_landmark_coordinates(_landmark)
        self.landmarks[_id] = (x_scaled, y_scaled)
        if config.annotate_imgs:
            # Write the landmark id on the image
            annotate_img(
                self.img,
                (x_scaled, y_scaled),
                str(_id),
            )

    def _get_scaled_landmark_coordinates(self, _landmark):
        height, width, channels = self.img.shape
        x_scaled = int(_landmark.x * width)
        y_scaled = int(_landmark.y * height)
        return x_scaled, y_scaled

    
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
        if config.allow_hand_plotting:
            plt.imshow(self.img)
            plt.title(
                f"Hand id {self.id}, boneage {self.boneage}, gender {self.gender}"
            )
            plt.show()
        else:
            return None
