import matplotlib.pyplot as plt


class Hand(object):
    def __init__(self, _image, _age, _gender, _id):
        self.img = _image
        self.age = _age
        self.gender = _gender
        self.id = _id
        self.raw_landmarks = None  # landmarks in the format given by mediapipe
        self.landmarks = None  # landmarks in a dictionary form more comfortable for us
        self.dict_centered_landmarks = None

        # Features for regression model
        self.ratio_length_finger_by_hand_width = None

    def standardize_contrast(self):
        pass
    
    def show(self):
        plt.imshow(self.img)
        plt.title(f"Hand id {self.id}")
        plt.show()
