import matplotlib.pyplot as plt
import config as config
import cv2
import math
from mediapipe_landmarks import MediapipeLandmarks
import numpy as np
import segmentations


class Hand(MediapipeLandmarks):
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
