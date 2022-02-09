from hand_processor import DataProcessor
from feature_extractor import Featurizer

if __name__ == "__main__":
    data_processor = DataProcessor()
    featurizer = Featurizer()
    data_processor.featurize_batch_of_hands()

    dp = data_processor
    for hand in data_processor.hands:
        featurizer.get_hand_landmarks(hand)
        featurizer.draw_landmarks(hand)
        featurizer.get_ratio_length_finger_by_hand_width(hand)
        featurizer.get_gap_bone_bones_proxy(hand)
        hand.show()
