import matplotlib.pyplot as plt
import config

class dotdict(dict):
    """Any class inheriting from this will be a dictionary whose attributes can be accessed with .dot notation"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Hand(dotdict):
    def __init__(self, _image, _age, _gender, _id):
        # Features for regression model
        for feature in config.FEATURE_NAMES:
            self[feature] = None
        
        self.img = _image
        self.age = _age
        self.gender = _gender
        self.id = _id
        self.raw_landmarks = None  # landmarks in the format given by mediapipe
        self.landmarks = None  # landmarks in a dictionary form more comfortable for us



    def standardize_contrast(self):
        pass

    def show(self):
        plt.imshow(self.img)
        plt.title(f"Hand id {self.id}, boneage {self.age}, gender {self.gender}")
        plt.show()
