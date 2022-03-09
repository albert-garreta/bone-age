import matplotlib.pyplot as plt
import config as config
import cv2
import math
import numpy as np
import segmentations as segmentations
import mediapipe as mp
from lib.utils import annotate_img
import config
from matplotlib.pyplot import figure

figure(figsize=(16, 12), dpi=80)

mp_hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class Hand(object):
    def __init__(self, _image, _boneage, _gender, _id, _segments):

        self.img = _image
        self.boneage = _boneage
        self.gender = _gender
        self.id = _id
        # list of 47 contours (contour = list of points) for each bone
        # if the bone does not exist, then there is None in its place
        self.segments = _segments

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
            # try:
            # print(feature_name)
            feature_value = eval(f"self.get_{feature_name}()")
            setattr(self, feature_name, feature_value)
        # except Exception as e:
        #    print(e)
        #    return False
        return True

    def get_boneage(self):
        return self.boneage

    def get_gender(self):
        return self.gender

    def get_metacarp_20_23_gap(self):
        """get the distance between bone 20 and 23 (see ref_hand.png)
        or between 21 and 22 when these bones exists"""
        return self._get_metacarp_gap(idx1=20, idx2=23)

    def get_metacarp_27_30_gap(self):
        return self._get_metacarp_gap(idx1=27, idx2=30)

    def get_metacarp_4_7_gap(self):
        return self._get_metacarp_gap(idx1=4, idx2=7)

    def get_metacarp_12_15_gap(self):
        return self._get_metacarp_gap(idx1=12, idx2=15)

    def _get_metacarp_gap(self, idx1, idx2):
        assert idx2 == idx1 + 3
        bone_idx1_contour = self.segments[idx1]
        bone_idx1_1_contour = self.segments[idx1 + 1]
        bone_idx2_1_contour = self.segments[idx2 - 1]
        bone_idx_2_contour = self.segments[idx2]
        first_bone_contour = (
            bone_idx1_1_contour if bone_idx1_1_contour else bone_idx1_contour
        )
        second_bone_contour = (
            bone_idx2_1_contour if bone_idx2_1_contour else bone_idx_2_contour
        )
        distance = segmentations.get_distance_between_contours(
            first_bone_contour, second_bone_contour
        )
        return distance

    def get_carp_bones_max_diameter(self):
        carp_bone_indices = range(37, 45)
        diameters = []
        for idx in carp_bone_indices:
            diameters.append(segmentations.get_diameter(self.segments[idx]))
        return np.max(diameters)

    def get_epifisis_diameter(self):
        epifisis_index = 45
        return segmentations.get_diameter(self.segments[epifisis_index])

    """----------------------------------------------------------------
    Get google's mediapipe's hand lanmarks methods
    https://google.github.io/mediapipe/solutions/hands
    ----------------------------------------------------------------"""

    def get_hand_landmarks(self):
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

    def draw_landmarks(self):
        if self.landmarks is None:
            print("No hand landmarks detected")
            return None
        mp_draw.draw_landmarks(
            self.img,
            self.raw_landmarks,
            # mp_hands.HAND_CONNECTIONS,
            list(mp.solutions.hands.HAND_CONNECTIONS),
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
        # for ldk_id, ldk in enumerate(self.landmarks):
        #     print(ldk)
        #     cv2.putText(
        #         self.img,
        #         str(ldk_id),
        #         (ldk[0], ldk[1]),
        #        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 255), 1
        #     )

    """----------------------------------------------------------------
    Organize segmentations
    The segmentations provided are unordered. Here we order them using 
    mediapipe's landmarks as reference
    ----------------------------------------------------------------"""

    def organize_segmentations(self):
        ldks = self.landmarks
        top_of_thumb = ldks[4]
        last_segment_id = None
        nonordered_segments = set(self.segments.keys())
        ordered_segment_ids = []
        while len(nonordered_segments) > 0:
            next_segment_id = self.find_next_segment(
                top_of_thumb, nonordered_segments, last_segment_id
            )
            ordered_segment_ids.append(next_segment_id)
            last_segment_id = next_segment_id
            if last_segment_id is not None:
                # print(nonordered_segments)
                nonordered_segments.remove(last_segment_id)
        self.ordered_segment_ids = ordered_segment_ids

    def find_next_segment(
        self, reference_landmark, nonordered_segment_ids, last_segment_id
    ):

        shortest_distance = np.inf
        next_segment = None
        if last_segment_id is not None:
            last_segment = self.segments[last_segment_id]
        else:
            last_segment = [reference_landmark]
        for segment in nonordered_segment_ids:
            distance = segmentations.get_distance_between_contours(
                last_segment, self.segments[segment]
            )
            if distance < shortest_distance:
                shortest_distance = distance
                next_segment = segment
        return next_segment

    """----------------------------------------------------------------
    Utility methods 
    ----------------------------------------------------------------"""

    def show(self):
        if config.allow_hand_plotting:
            plt.imshow(self.img)
            plt.title(
                f"Hand id {self.id}, boneage {self.boneage}, gender {self.gender}"
            )
            plt.show()
        else:
            return None
