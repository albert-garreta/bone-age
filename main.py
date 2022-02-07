from data_processor import DataProcessor
from feature_extractor import Featurizer

if __name__ == "__main__":
    data_processor = DataProcessor()
    featurizer = Featurizer()
    data_processor.load_batch_of_hands()

    dp = data_processor
    hand = data_processor.hands[0]
    featurizer.get_hand_landmarks(hand)
    featurizer.draw_landmarks(hand)
    featurizer.get_ratio_length_finger_by_hand_width(hand)
    hand.show()