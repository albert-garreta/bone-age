import matplotlib.pyplot as plt
import sys

sys.path.append("../bone-age")
import config
import cv2
import math
import os
import numpy as np
import lib.segmentations as segmentations
import mediapipe as mp
from lib.utils import (
    annotate_img,
    euclidean_distance,
)
from lib.hand_utils import get_consecutive_ldk_distances
from matplotlib.pyplot import figure

figure(figsize=(12, 8), dpi=80)
mp_hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

COLOR_TO_BGR = {
    "green": (0, 255, 0),
    "blue": (255, 0, 0),
    "red": (0, 0, 255),
    "cyan": (255, 255, 0),
    "purple": (255, 0, 255),
    "yellow": (0, 255, 255),
}


class Hand(object):
    def __init__(self, _image, _boneage, _gender, _id, _segments):

        self.img = _image
        self.boneage = _boneage
        self.gender = _gender
        self.id = _id
        # list of contours (contour = segment=  list of points)
        self.segments = _segments

    def __repr__(self):
        return f"Hand id {self.id}, age (years) {self.boneage/12}, gender {self.gender}"

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
                feature_value = eval(f"self.get_{feature_name}()")
                setattr(self, feature_name, feature_value)
            except Exception as e:
               print(f"Exception encountered when creating feature {feature_name}")
               print(e)
               print(self)
               return False, feature_name
        return True, None

    def get_id(self):
        return int(self.id)

    def get_boneage(self):
        return self.boneage

    def get_gender(self):
        return self.gender

    def get_gap_ratio(self, landmark_num):
        """landmark_num = google's mediapipe landmark where the gap between bones is measured"""
        distance = self.get_gap(landmark_num)
        if landmark_num == 9:
            landmarks_for_calc_vertical_length = [0, 9, 10, 11, 12]
        if landmark_num == 5:
            landmarks_for_calc_vertical_length = [0, 5, 6, 7, 8]
        if landmark_num == 13:
            landmarks_for_calc_vertical_length = [0, 13, 14, 15, 16]
        if landmark_num == 17:
            landmarks_for_calc_vertical_length = [0, 17, 18, 19, 20]
        if landmark_num == 10:
            landmarks_for_calc_vertical_length = [0, 9, 10, 11, 12]
        if landmark_num == 6:
            landmarks_for_calc_vertical_length = [0, 5, 6, 7, 8]
        if landmark_num == 14:
            landmarks_for_calc_vertical_length = [0, 13, 14, 15, 16]
        if landmark_num == 18:
            landmarks_for_calc_vertical_length = [0, 17, 18, 19, 20]
        vertical_length = get_consecutive_ldk_distances(
            self.landmarks,
            landmarks_for_calc_vertical_length
        )
        return distance / max(0.1, vertical_length)

    def get_gap_5(self):
        return self.get_gap(5)
    def get_gap_6(self):
        return self.get_gap(6)
    def get_gap_9(self):
        return self.get_gap(9)
    def get_gap_10(self):
        return self.get_gap(10)
    def get_gap_13(self):
        return self.get_gap(13)
    def get_gap_14(self):
        return self.get_gap(14)
    def get_gap_17(self):
        return self.get_gap(17)
    def get_gap_18(self):
        return self.get_gap(18)
    
    def get_gap_ratio_5(self):
        return self.get_gap_ratio(5)
    def get_gap_ratio_6(self):
        return self.get_gap_ratio(6)
    def get_gap_ratio_9(self):
        return self.get_gap_ratio(9)
    def get_gap_ratio_10(self):
        return self.get_gap_ratio(10)
    def get_gap_ratio_13(self):
        return self.get_gap_ratio(13)
    def get_gap_ratio_14(self):
        return self.get_gap_ratio(14)
    def get_gap_ratio_17(self):
        return self.get_gap_ratio(17)
    def get_gap_ratio_18(self):
        return self.get_gap_ratio(18)
    
    def get_gap(self, landmark_num):
        """landmark_num = google's mediapipe landmark where the gap between bones is measured"""
        constraints = [
            self.get_point_in_between(self.landmarks[9], self.landmarks[13]),
            self.get_point_in_between(self.landmarks[9], self.landmarks[5]),
        ]
        return self._get_dist_to_point_of_centroid_closest_two_segments(
            self.landmarks[landmark_num], constraints
        )

    def get_constraints_for_calculaiton_of_gaps(self, landmark_num):
        if landmark_num == 9:
            constraints = [
                self.get_point_in_between(self.landmarks[9], self.landmarks[13]),
                self.get_point_in_between(self.landmarks[9], self.landmarks[5]),
            ]
        elif landmark_num == 5:
            constraints = [
                self.get_point_in_between(self.landmarks[5], self.landmarks[9]),
                None,
            ]
        elif landmark_num == 13:
            constraints = [
                self.get_point_in_between(self.landmarks[13], self.landmarks[17]),
                self.get_point_in_between(self.landmarks[13], self.landmarks[9]),
            ]
        elif landmark_num == 17:
            constraints = [
                None,
                self.get_point_in_between(self.landmarks[17], self.landmarks[13]),
            ]
        elif landmark_num == 6:
            constraints = [
                self.get_point_in_between(self.landmarks[6], self.landmarks[10]),
                None,
            ]
        elif landmark_num == 10:
            constraints = [
                self.get_point_in_between(self.landmarks[10], self.landmarks[14]),
                self.get_point_in_between(self.landmarks[10], self.landmarks[6]),
            ]
        elif landmark_num == 14:
            constraints = [
                self.get_point_in_between(self.landmarks[14], self.landmarks[18]),
                self.get_point_in_between(self.landmarks[14], self.landmarks[10]),
            ]
        elif landmark_num == 18:
            constraints = [
                None,
                self.get_point_in_between(self.landmarks[18], self.landmarks[14]),
            ]
        else:
            raise Exception(
                f"Computing `gap_{landmark_num}`: landmark number is not supported"
            )
        return constraints

    @staticmethod
    def get_point_in_between(point1, point2, k=0.2):
        return (
            (k * point1[0] + (1 - k) * point2[0]),
            (k * point1[1] + (1 - k) * point2[1]),
        )

    @staticmethod
    def get_segment_centroid(segment):
        center_x = int(np.mean([p[0] for p in segment]))
        center_y = int(np.mean([p[1] for p in segment]))
        return (center_x, center_y)

    def _get_dist_to_point_of_centroid_closest_two_segments(
        self, point, constraints=None
    ):
        valid_segments = {}
        if constraints is None:
            valid_segments = self.segments
        else:
            for seg_id, segment in self.segments.items():
                # WARNING: !!! fist compoment corresponds to the y-axis of an array!!
                x_coords = [p[0] for p in segment]
                y_coords = [p[1] for p in segment]
                x_centroid, y_centroid = self.get_segment_centroid(segment)
                if constraints[0] is not None and constraints[1] is not None:
                    if (
                        max(x_coords) < constraints[1][0]
                        and min(x_coords) > constraints[0][0]
                    ):
                        valid_segments[seg_id] = segment
                if constraints[0] is None and constraints[1] is not None:
                    if max(x_coords) < constraints[1][0]:
                        valid_segments[seg_id] = segment
                if constraints[0] is not None and constraints[1] is None:
                    if min(x_coords) > constraints[0][0]:
                        valid_segments[seg_id] = segment
        distances = [
            (segment_id, euclidean_distance(point, self.get_segment_centroid(segment)))
            for segment_id, segment in valid_segments.items()
        ]
        distances.sort(key=lambda x: x[1])
        closest_segment1 = self.segments[distances[0][0]]
        closest_segment2 = self.segments[distances[1][0]]

        annotate_img(self.img, point, "A")
        segmentations.draw_all_contours(
            self.img,
            {1: closest_segment1, 2: closest_segment2},
            color=(255, 0, 255),
            write_contour_number=True,
        )

        gap_length = segmentations.get_distance_between_two_lists_of_points(
            closest_segment1, closest_segment2
        )
        return gap_length

    @staticmethod
    def get_min_y_coord(list_of_points):  #
        return [p[1] for p in list_of_points]

    def get_distance_to_ratio_by(self):
        return get_consecutive_ldk_distances(self.landmarks, [0, 5, 6, 7, 8])

    def get_carp_bones_max_distances(self):
        self.green_segments = self.detect_color_segments(self.segments, "green")
        green_seg_list = list(self.green_segments.values())
        if len(green_seg_list) > 0:
            green_seg_pair_list = [
                (seg1, seg2)
                for idx, seg1 in enumerate(green_seg_list)
                for seg2 in green_seg_list[idx:]
            ]
            distances = []
            for segment_pair in green_seg_pair_list:
                distances.append(
                    segmentations.get_distance_between_two_lists_of_points(*segment_pair)
                )
            carp_bones_max_distances = max(distances)
        else:
            carp_bones_max_distances = 0
        self.carp_bones_max_distances = carp_bones_max_distances
        return carp_bones_max_distances

    def get_carp_bones_max_distances_ratio(self):
        return self.carp_bones_max_distances / self.get_distance_to_ratio_by()

    def get_carp_bones_max_diameter(self):
        if len(self.green_segments) > 0:
            carp_bones_max_diameter = max(
                [
                    segmentations.get_diameter(segment)
                    for segment in self.green_segments.values()
                ]
            )
        else:
            carp_bones_max_diameter = 0
        self.carp_bones_max_diameter = carp_bones_max_diameter
        return carp_bones_max_diameter

    def get_carp_bones_sum_perimeters(self):
        if len(self.green_segments) > 0:
            carp_bones_sum_perimeters = np.sum(
                [
                    segmentations.get_perimeter(segment)
                    for segment in self.green_segments.values()
                ]
            )
        else:
            carp_bones_sum_perimeters = 0
        self.carp_bones_sum_perimeters = carp_bones_sum_perimeters
        return carp_bones_sum_perimeters

    def get_yellow_sum_perimeters(self):
        self.yellow_segments = self.detect_color_segments(self.segments, "yellow")

        if len(self.yellow_segments) > 0:
            yellow_sum_perimeters = np.sum(
                [
                    segmentations.get_perimeter(segment)
                    for segment in self.yellow_segments.values()
                ]
            )
        else:
            yellow_sum_perimeters = 0
        self.yellow_sum_perimeters = yellow_sum_perimeters
        return yellow_sum_perimeters

    def get_yellow_ratio_green(self):
        return self.carp_bones_sum_perimeters / self.yellow_sum_perimeters

    def get_carp_bones_sum_perimeters_ratio(self):
        return self.carp_bones_sum_perimeters / self.get_distance_to_ratio_by()

    def get_max_purple_diameter(self):
        self.purple_segments = self.detect_color_segments(self.segments, "purple")

        if len(self.purple_segments) > 0:
            max_purple_diameter = max(
                [
                    segmentations.get_diameter(segment)
                    for segment in self.purple_segments.values()
                ]
            )
        else:
            max_purple_diameter = 0
        self.max_purple_diameter = max_purple_diameter
        return max_purple_diameter

    def get_max_purple_diameter_ratio(self):
        return self.max_purple_diameter / self.get_distance_to_ratio_by()

    def get_carp_bones_max_diameter_ratio(self):
        return self.carp_bones_max_diameter / self.get_distance_to_ratio_by()

    def get_epifisis_max_diameter(self):
        self.red_segments = self.detect_color_segments(self.segments, "red")

        if len(self.red_segments) > 0:
            epifisis_max_diameter = max(
                [
                    segmentations.get_diameter(segment)
                    for segment in self.red_segments.values()
                ]
            )
        else:
            epifisis_max_diameter = 0
        self.epifisis_max_diameter = epifisis_max_diameter
        return epifisis_max_diameter

    def get_epifisis_max_diameter_ratio(self):
        return self.epifisis_max_diameter / self.get_distance_to_ratio_by()

    def detect_color_segments(self, segments, color):
        """
        Among the segments in `self.segments`, it finds those of the color specified
        as colored in the corresponding hand file in the folder data_tagged
        (yellow = phalanx and metaphalanx, green = wrist bones,
        red = the two bones above the two arm bones,
        purple = bones between phalanx and metaphalanxs)
        """
        color_segments = {}
        colored_img = cv2.imread(
            os.path.join(config.colored_data_dir, f"{self.id}.png")
        )
        for segment_id, segment in segments.items():
            if segmentations.has_color(colored_img, segment, color):
                color_segments[segment_id] = segment
        self.img = segmentations.draw_all_contours(
            self.img, color_segments, color=COLOR_TO_BGR[color]
        )
        return color_segments

    """----------------------------------------------------------------
    Get google's mediapipe's hand lanmarks methods
    https://google.github.io/mediapipe/solutions/hands
    ----------------------------------------------------------------"""

    def get_hand_landmarks(self):
        """Creates the attribute `landmarks`: a dictionary with items of the form:
        landmark_id (int) : (x_coordinate, y_coordinate)
        """
        self.landmarks = None
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
        return True

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
            self._process_individual_landmark(id, landmark)

    def _process_individual_landmark(self, _id, _landmark):
        x_scaled, y_scaled = self._get_scaled_landmark_coordinates(_landmark)
        self.landmarks[_id] = (x_scaled, y_scaled)
        if config.make_drawings:
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
            list(mp.solutions.hands.HAND_CONNECTIONS),
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
        for ldk_id, ldk in self.landmarks.items():
            cv2.putText(
                self.img,
                str(ldk_id),
                (ldk[0], ldk[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 0, 255),
                1,
            )

    """----------------------------------------------------------------
    Utility methods 
    ----------------------------------------------------------------"""

    def show(self):
        if config.allow_hand_plotting:
            plt.imshow(self.img)
            plt.title(
                f"Hand id {self.id}"  # , boneage {self.boneage}, gender {self.gender}"
            )
            plt.show()
        else:
            return None

    """----------------------------------------------------------------
    **NOT USED**
    
    Organize segmentations
    The segmentations provided are unordered. Here we order them using 
    mediapipe's landmarks as reference
    ----------------------------------------------------------------"""
    """
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
    """
