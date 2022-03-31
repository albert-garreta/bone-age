#from feature_extractor import Featurizer
import sys
sys.path.append("../bone-age")

from lib.hand_factory import HandFactory

if __name__ == "__main__":
    data_processor = HandFactory()
    #featurizer = Featurizer()
    data_processor.get_features_dataframe()
