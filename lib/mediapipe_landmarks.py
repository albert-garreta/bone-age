import mediapipe as mp
from utils import annotate_img
from hand import Hand
import config

mp_hands = mp.solutions.hands.Hands()
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


class MediapipeLandmarks(object):
    """----------------------------------------------------------------
    Get google's mediapipe's hand lanmarks methods
    https://google.github.io/mediapipe/solutions/hands
    ----------------------------------------------------------------"""

    def __init__(self, hand: Hand):
        """
        landmarks: dictionary mapping integers (landmark number) to a point in self.img
        """
        self.landmarks = {}
        self.hand = hand

    def get_hand_landmarks(self):
        """Creates the attribute `landmarks`: a dictionary with items of the form:
        landmark_id (int) : (x_coordinate, y_coordinate)
        """
        raw_landmarks, success = self._get_raw_landmarks()
        if not success:
            print(
                f"No hand landmarks were found in Hand {self.hand.id} "
                f"with boneage {self.hand.boneage} and gender {self.hand.gender}"
            )
            return None
        else:
            self.raw_landmarks = raw_landmarks
            self._convert_raw_landmarks()

    def _get_raw_landmarks(self):
        mp_result = mp_hands.process(self.hand.img)
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
                self.hand.img,
                (x_scaled, y_scaled),
                str(_id),
            )

    def _get_scaled_landmark_coordinates(self, _landmark):
        height, width, channels = self.hand.img.shape
        x_scaled = int(_landmark.x * width)
        y_scaled = int(_landmark.y * height)
        return x_scaled, y_scaled

    def draw_landmarks(self):
        if self.hand.raw_landmarks is None:
            print("No hand landmarks detected")
            return None
        mp_draw.draw_landmarks(
            self.hand.img,
            self.hand.raw_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )
