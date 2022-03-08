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

"""TODO: STUFF TO TRY
    - Separate in age ranges as in BoneXpert
"""


class HandInterface(object):
    # TODO: how do interfaces work really?
    def __init__(self):
        for feature in config.ALL_FEATURE_NAMES:
            setattr(self, feature, None)


class Hand(HandInterface):
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
        self.length_middle_finger = None
        self._gap_proxy_mean = None
        self._gap_proxy_std = None

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

    # @no_landmarks_wrapper
    def featurize(self) -> bool:
        """Main function to create the features.
        The features used are customizable from the config file.
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
        # These are redundant but we need them for our customizable features logic
        return self.boneage

    def get_gender(self):
        # These are redundant but we need them for our customizable features logic
        return self.gender

    @no_landmarks_wrapper
    def get_length_10_9(self):
        point1 = self.landmarks[10]
        point2 = self.landmarks[9]
        line_function = get_line_function(point1, point2)
        pass

    @no_landmarks_wrapper
    def get_length_middle_finger(self):
        middle_finger_landmark_ids = [12, 11, 10, 9]
        return get_consecutive_ldk_distances(self.landmarks, middle_finger_landmark_ids)

    @no_landmarks_wrapper
    def get_length_top_palm(self):
        top_palm_landmark_ids = [17, 13, 9, 5]
        return get_consecutive_ldk_distances(self.landmarks, top_palm_landmark_ids)

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
        # square = cv2.adaptiveThreshold(
        #     square, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2
        # )
        square = cv2.equalizeHist(square)
        # plt.imshow(square)
        # plt.show()
        square = cv2.cvtColor(square, cv2.COLOR_GRAY2RGB)

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
        return self.get_length_middle_finger()  / self._gap_proxy_std

    def get_ratio_finger_to_gap_mean(self):
        self._get_hand_landmarks()
        #self.get_length_middle_finger()
        self.get_gap_bones_proxy()
        if self.landmarks:
            return self.get_length_middle_finger() / self._gap_proxy_mean
        else:
            return None
    @no_landmarks_wrapper    
    def get_carp_bones_area_ratio(self):
        # print('hello')
        self._fill_in_connected_comp_info()
        #return self._get_carp_bones_area() / max(0.1, self._get_all_bones_area())
        return np.sqrt(self._get_carp_bones_area()) / self.get_length_middle_finger()
    
    
    @no_landmarks_wrapper
    def get_epifisis_area_ratio(self):
        # print('hello')
        #self._fill_in_connected_comp_info()
        #return self._get_carp_bones_area() / max(0.1, self._get_all_bones_area())
        return np.sqrt(self._get_epifisis_area()) / self.get_length_middle_finger()

    def _get_epifisis_area(self):
        epifisis_connected_comp_info = self.connected_components_by_color["red"]
        return self._get_color_area(epifisis_connected_comp_info)

    @staticmethod
    def first_nonzero(arr, axis, invalid_val=-1):
        mask = arr!=0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)
    @staticmethod
    def last_nonzero(arr, axis, invalid_val=-1):
        mask = arr!=0
        return np.where(mask.any(axis=axis), mask.argmin(axis=axis), invalid_val)

    def get_carp_bones_area_ratio2(self):
        # print('hello')
        img = self.segmented_img
        plt.imshow(img)
        plt.show()
        img_hsv = cv2.cvtColor(self.segmented_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, lowerb = (0,1,0), upperb = (255,255,255))
        w, h = mask.shape
       
        # smallest column containing a non-zero value, etc
        first_nonzero_col = self.first_nonzero(mask, axis=1)
        first_nonzero_row = self.first_nonzero(mask, axis=0)
        last_nonzero_col = self.last_nonzero(mask, axis=1)
        last_nonzero_row = self.last_nonzero(mask, axis=0)
        
        print(first_nonzero_col)
        print(first_nonzero_row)
        print(last_nonzero_col)
        print(last_nonzero_row)
        total_area = (last_nonzero_col-first_nonzero_col)*(last_nonzero_row-first_nonzero_row)
        
        #green_mask = cv2.inRange(self.segmented_img, lowerb = (105, 1, 0), upperb= (135, 255, 255))
        green_mask = cv2.inRange(self.segmented_img, lowerb =(115, 185, 115), upperb =(170, 255, 185))
        plt.imshow(img_hsv)
        plt.show()

        plt.imshow(green_mask)
        plt.show()
        
        w, h = green_mask.shape
        smallest_col = np.argmin([(mask>0)[:,row].any() for row in range(h)])
        smallest_row = np.argmin([(mask>0)[col,:].any() for col in range(w)])
        largest_col = np.argmin([(mask>0)[:,row].any() for row in range(h)])
        largest_row = np.argmin([(mask>0)[col,:].any() for col in range(w)])
        
        total_carp_area = (largest_col-smallest_col)*(largest_row-smallest_row)
        
        return np.sqrt(total_carp_area/total_area)
        
    def _fill_in_connected_comp_info(self):
        # plt.imshow(self.img)
        # plt.show()
        self.connected_components_by_color = {}
        BGR_color_bounds = {
            "yellow": {"lower": (0, 170, 170), "upper": (50, 255, 255)},
            "green": {"lower": (115, 185, 115), "upper": (170, 255, 185)},
            "red": {"lower": (0, 0, 180), "upper": (100, 100, 255)},
            "purple": {"lower": (140, 0, 140), "upper": (210, 75, 205)},
            "cyan": {"lower": (200, 185, 0), "upper": (255, 255, 65)},
        }
        for color, color_bounds in BGR_color_bounds.items():
            # print(color)
            lowerb, upperb = (
                color_bounds["lower"],
                color_bounds["upper"],
            )
            # plt.imshow(self.img)
            # plt.show()
            mask = cv2.inRange(self.segmented_img, lowerb=lowerb, upperb=upperb)

            # plt.imshow(mask)
            # plt.show()
            _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
            hand_masked = cv2.bitwise_and(self.img, self.img, mask=mask)
            hand_masked = cv2.cvtColor(hand_masked, cv2.COLOR_BGR2GRAY)
            hand_masked = cv2.threshold(
                hand_masked, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            )[1]
            # plt.imshow(hand_masked)
            # plt.show()
            hand_masked = hand_masked.astype(np.uint8)

            cc_output = cv2.connectedComponentsWithStats(
                hand_masked, 4, cv2.CV_8S  # connectivity  4 or 8
            )
            (num_labels, labels, stats, centroids) = cc_output
            self.connected_components_by_color[color] = {
                "num_labels": num_labels,
                "labels": labels,
                "stats": stats,
                "centroids": centroids,
            }

    def _get_all_bones_area(self):
        total_area = 0
        for color, cc_color_info in self.connected_components_by_color.items():
            total_area += self._get_color_area(cc_color_info)
            # print(color, total_area)
        return total_area

    @staticmethod
    def _get_color_area(connected_components):
        num_labels, stats = (
            connected_components["num_labels"],
            connected_components["stats"],
        )
        # First two connected components correspondo to the background and
        # and another one I'm not sure where it comes from, but it is
        # (as far as I can tell) a bounding box for all the components
        # (sometimes this box is not tight)
        total_color_area = sum(
            [stats[label, cv2.CC_STAT_AREA] for label in range(2, num_labels)]
        )
        return total_color_area

    def _get_carp_bones_area(self):
        carp_bones_connected_comp_info = self.connected_components_by_color["green"]
        return self._get_color_area(carp_bones_connected_comp_info)
    
    
    
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
