import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BGR_color_bounds = {
    "yellow": {"lower": (0, 180, 180), "upper": (40, 255, 255)},
    "green": {"lower": (125, 195, 125), "upper": (166, 255, 176)},
    "red": {"lower": (0, 0, 190), "upper": (90, 90, 255)},
    "purple": {"lower": (151, 16, 150), "upper": (200, 62, 191)},
    "cyan": {"lower": (210, 195, 0), "upper": (255, 255, 55)},
}

feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

if __name__ == "__main__":
    dir = "data/data_tagged"
    imgs = os.listdir(dir)
    for img in imgs[4:5]:
        try:
            hand = cv2.imread(os.path.join(dir, img), 1)
            print(hand.shape)
            for color, bounds in BGR_color_bounds.items():
                lowerb, upperb = bounds["lower"], bounds["upper"]
                mask = cv2.inRange(hand, lowerb=lowerb, upperb=upperb)
                _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
                hand_masked = cv2.bitwise_and(hand,hand,mask=mask)

                # print(mask.shape)
                # mask = mask.reshape(*mask.shape, 1)
                # corners = cv2.goodFeaturesToTrack(mask, **feature_params)
                # if corners is not None:
                #     for x, y in np.float32(corners).reshape(-1, 2):
                #         cv2.circle(a, (x, y), 10, (0, 255, 0), 1)

                plt.imshow(mask)
                plt.title(f"{img}-{color}")
                plt.show()
                plt.imshow(hand_masked)
                plt.show()
        except Exception as e:
            print(e)
