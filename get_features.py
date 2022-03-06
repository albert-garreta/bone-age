#from feature_extractor import Featurizer
from classes.hand_factory import HandFactory

if __name__ == "__main__":
    data_processor = HandFactory()
    #featurizer = Featurizer()
    data_processor.featurize_batch_of_hands()
