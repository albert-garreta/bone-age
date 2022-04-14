import matplotlib.pyplot as plt
import sys

from sklearn import get_config
from sympy import get_contraction_structure

sys.path.append("../bone-age")
import config
import cv2
import math
import os
import numpy as np
import lib.segmentations as segmentations
import mediapipe as mp
from lib.utils import annotate_img, euclidean_distance, COLOR_NAME_TO_BGR
from matplotlib.pyplot import figure

figure(figsize=(12, 8), dpi=80)

# Google mediapipe methods
mp_hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class Hand(object):
    def __init__(self, _image, _boneage, _gender, _id, _segments):
        self.img = _image
        self.boneage = _boneage
        self.gender = _gender
        self.id = _id
        # list of contours (contour =  list of points). A contour sometimes is called
        # `segment` in this repository
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

    """----------------------------------------------------------------
    Methods for creating features related to the gaps between bones, i.e.
    gap_bone_ratio_<LANDMARK_ID> and gab_bone_<LANDMARK>
    ----------------------------------------------------------------"""

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

    def get_gap_ratio(self, landmark_num):
        """landmark_num = google's mediapipe landmark where the gap between bones is measured.
        See `landmarks_example.png` for a reference on what each number indicates"""
        distance = self.get_gap(landmark_num)

        # The variable `landmaks_for_calc_vertical_length` contains the landmarks for which
        # we compute the sum of distances between each two consecutive landmarks in the list.
        # Then, we quotient the gap value `distance` by such sum of distances, in order to obtain
        # features of the type gap_ratio_<landmark_num>.

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
        vertical_length = self.get_consecutive_ldk_distances(
            self.landmarks, landmarks_for_calc_vertical_length
        )
        return distance / max(0.1, vertical_length)

    def get_gap(self, landmark_num):
        """
        landmark_num = google's mediapipe landmark where the gap between bones is measured

        This function should return the distance between two bones. The two bones are determined
        by `landmark_num`. For example, if `landmark_num=5`, then this function
        should return the distance between the lower phalanx of the index finger, and the metacarpal
        of the index finger (see the `lanmarks_example.png` to see where the landmark 5 is placed and
        what "gap_distance" corresponds to such landmark).


        ABOUT BONE FORMATIONS:

        Sometimes phalanxes and metacarpals are accompanied of "bone formations"
        (these are small bones that appear as the person grows and which are eventually
        merged into the "normal" bones). For examples: the bone formations are contoured
        in purple color in the images from `tagged_data_colored_contours`.

        In case the phalanx and the metacarpal we are focusing on have bone formations, then
        when computing the distance between these bones, we include the bone formations as part
        of the phalanx and metacarpal, respectively.


        HOW TO FIND THIS FEATURE:

        The way to compute this feature is not super easy: naively, we could just take the two
        contours which are closer to `landmark_num`. However, in many cases this will detect
        a phalanx and its "bone fromation" (the distance between these two bones is much smaller
        than the actual distance we want to find). Hence, we need to refine this method a little.

        To fix this I added the following heuristic which seems to work reasonably, BUT IS NOT OPTIMAL
        BY ANY MEANS: we take the landmarks A and B such that A is the closest landmark above `landmark_num`
        and B is the closest landmark below `landmark_num`. Then we take the points M_1, M_2
        in the segments [A, landmark_num] and [landmark_num, B] (we take the points that are 80% close to A or B,
        respectively) and then we set as invalid all contours
        that have some pixel above or below (vertically) M_1 or M_2.

        The hope is that this discards the pathological case mentioned above: typically, the phalanx
        has some pixel above M_1, and similarly for the metacarpal.

        TODO: this is something to be improved

        """
        constraints = self.get_constraints_for_calculaiton_of_gaps(landmark_num)
        return self.get_distance_between_bones_close_to_landmark(
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
    def get_segment_mean_point(segment):
        center_x = int(np.mean([p[0] for p in segment]))
        center_y = int(np.mean([p[1] for p in segment]))
        return (center_x, center_y)

    #
    #
    #   Here we have methods for finding the gap in the gap variables.
    #
    #

    def get_gap_between_bones_close_to_landmark(self, landmark, constraints=None):
        """See the documentation of `get_gap`"""

        # Here we restrict to contours (also called segments) that have no pixels
        # above or below the two points given in `constraints`.
        # The valid contours are stored in `valid_segments`

        # `constraints[0]` is the point that marks the lower bound, while
        # `constraints[1]` marks the upper bound.

        valid_segments = self.get_valid_segments_for_gap_calculations(
            landmark, constraints
        )

        # Now we get the two segments from valid_segments that are closer to the landmark
        closest_segment1, closest_segment2 = self.get_two_closest_segments_to_point(
            landmark, valid_segments
        )

        # Compute the distance between the two found segments
        gap_length = segmentations.get_distance_between_two_lists_of_points(
            closest_segment1, closest_segment2
        )

        # This will draw the two selected segments
        if config.make_drawings:
            annotate_img(self.img, landmark, "A")
            segmentations.draw_all_contours(
                self.img,
                {1: closest_segment1, 2: closest_segment2},
                color=(255, 0, 255),
                write_contour_number=True,
            )

        return gap_length

    def get_valid_segments_for_gap_calculations(self, landmark, constraints):
        # See the documentation for `get_gap_between_bones_close_to_landmark`
        if constraints is None:
            valid_segments = self.segments
        else:
            valid_segments = {}
            for seg_id, segment in self.segments.items():
                x_coords = [point[0] for point in segment]
                if constraints[0] is not None and constraints[1] is not None:
                    if (
                        max(x_coords) < constraints[1][0]
                        and min(x_coords) > constraints[0][0]
                    ):
                        valid_segments[seg_id] = segment
                # NOTE: Currently the next two cases are never used.
                if constraints[0] is None and constraints[1] is not None:
                    if max(x_coords) < constraints[1][0]:
                        valid_segments[seg_id] = segment
                if constraints[0] is not None and constraints[1] is None:
                    if min(x_coords) > constraints[0][0]:
                        valid_segments[seg_id] = segment
        return valid_segments

    def get_two_closest_segments_to_point(self, point, segments):
        """Find the closest two segments among `segments` to a
        # given `point`.

        Args:
            point: (int, int)
            segments: dictionary of the form {segment_id: segment}. A segment is a
                    list of points

        Returns:
            (closest_segment1, closest_segment2)
        """
        #
        ids_and_distances = [
            (
                segment_id,
                euclidean_distance(point, self.get_segment_mean_point(segment)),
            )
            for segment_id, segment in segments.items()
        ]
        ids_and_distances.sort(key=lambda x: x[1])
        closest_segment1 = segments[ids_and_distances[0][0]]
        closest_segment2 = segments[ids_and_distances[1][0]]
        return closest_segment1, closest_segment2

    """----------------------------------------------------------------
    Methods for creating features related to the diameter of bones
    ----------------------------------------------------------------"""

    def get_distance_to_ratio_by(self):
        return self.get_consecutive_ldk_distances(self.landmarks, [0, 5, 6, 7, 8])

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
                    segmentations.get_distance_between_two_lists_of_points(
                        *segment_pair
                    )
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
            self.img, color_segments, color=COLOR_NAME_TO_BGR[color]
        )
        return color_segments

    """----------------------------------------------------------------
    Miscellaneous
    ----------------------------------------------------------------"""

    @staticmethod
    def get_distance(point1, point2):
        # point = tuple(int, int)
        translated_point = (point1[0] - point2[0], point1[1] - point2[1])
        return np.sqrt(translated_point[0] ** 2 + translated_point[1] ** 2)

    def get_consecutive_ldk_distances(self, _dict_landmarks, _landmark_ids_list):
        total_distance = 0
        for idx, ldk_id in enumerate(_landmark_ids_list[:-1]):
            next_ldk_id = _landmark_ids_list[idx + 1]
            distance = self.get_distance(
                _dict_landmarks[ldk_id], _dict_landmarks[next_ldk_id]
            )
            # print(distance, ldk_id, next_ldk_id)
            total_distance += distance
        return total_distance

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
        plt.imshow(self.img)
        plt.title(
            f"Hand id {self.id}"  # , boneage {self.boneage}, gender {self.gender}"
        )
        plt.show()
