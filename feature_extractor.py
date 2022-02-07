import cv2
import mediapipe as mp
import numpy as np
from utils import annotate_img


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

#
# Use AGE as a feature!
class Featurizer(object):
    def __init__(
        self,
    ):
        self.mp_hands = mp_hands.Hands()

    def get_hand_landmarks(self, _hand):
        # Returns a dictionary with items of the form: 
        #       landmark_id (int) : (x_coordinate, y_coordinate)
        
        dict_centered_landmarks = {}
        # hand is an instance of the Hand class
        img = _hand.img
        mp_result = self.mp_hands.process(img)
        # mp_result.multi_hand_landmarks is None if no landmarks were found
        # Otherwise it is a list (?) with length the number of hands detected
        # each entry is a list of a particular hand landmark
        if mp_result.multi_hand_landmarks is None:
            print(f"Hand id {_hand.id}: No hand landmarks were found")
            return None
        
        # We store the hand_landmark data structure given my mediapipe as `raw_landmarks`
        # This is only used so far in the `draw_landmarks` method
        # We will be working with the dictionary version that we create now
        # The zero is because there will always be only one hand
        _hand.raw_landmarks = mp_result.multi_hand_landmarks[0]
        landmarks = mp_result.multi_hand_landmarks[0].landmark
        for id, landmark in enumerate(landmarks):
            # Here we scale the point to match the img size
            height, width, channels = _hand.img.shape
            x_img = int(landmark.x * width)
            y_img = int(landmark.y * height)

            # Now we add the point to the dictionary
            dict_centered_landmarks[id] = (x_img, y_img)
            
            # And we write the landmark id on the image
            annotate_img(
                _hand.img,
                (x_img, y_img),
                str(id),
            )
        _hand.landmarks = dict_centered_landmarks

    def get_ratio_length_finger_by_hand_width(self, _hand):
        assert _hand.landmarks is not None
        dict_landmarks = _hand.landmarks
        # tip to palm
        middle_finger_landmark_ids = [12, 11, 10, 9]
        # left to right
        top_palm_landmark_ids = [17, 13, 9, 5]
        middle_finger_length = get_consecutive_ldk_distances(
            dict_landmarks, middle_finger_landmark_ids
        )
        top_palm_length = get_consecutive_ldk_distances(
            dict_landmarks, top_palm_landmark_ids
        )
        ratio = middle_finger_length / top_palm_length
        self.ratio_length_finger_by_hand_width = ratio
        txt1 = f"middle finger length: {middle_finger_length}"
        txt2 = f"top palm_width: {top_palm_length}"
        txt3 = f"ratio finger palm: {ratio}"
        annotate_img(_hand.img, (0, 1600), f"finger_to_palm {round(ratio,2)}")
        print(f"Hand id: {_hand.id}")
        print(txt1)
        print(txt2)
        print(txt3)
        return ratio
    
    def get_gap_bone_gap_proxy(self):
        pass

    def draw_landmarks(self, _hand):
        assert _hand.raw_landmarks is not None, "No hand landmarks detected"
        mp_draw.draw_landmarks(
            _hand.img,
            _hand.raw_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )


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
