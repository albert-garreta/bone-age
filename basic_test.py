from classes.hand_factory import HandFactory
from classes.hand_utils import align_batch_of_hands
from scripts.data_analysis import main as do_data_analysis
if __name__ == "__main__":
    hand_factory = HandFactory()
    hand_factory.featurize_batch_of_hands()

    for hand in hand_factory.hands:
        hand.show()
        
    do_data_analysis()
    
    # align_batch_of_hands(hand_factory.hands)
    
    