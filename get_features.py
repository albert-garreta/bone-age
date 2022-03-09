#from feature_extractor import Featurizer
from lib.hand_factory import HandFactory

if __name__ == "__main__":
    data_processor = HandFactory()
    #featurizer = Featurizer()
    data_processor.get_features_dataframe()
