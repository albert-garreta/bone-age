from hand_processor import DataProcessor
#from feature_extractor import Featurizer

if __name__ == "__main__":
    data_processor = DataProcessor()
    #featurizer = Featurizer()
    data_processor.featurize_batch_of_hands()
