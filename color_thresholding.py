import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BGR_color_bounds = {
    "yellow": {"lower": (0, 170, 170), "upper": (50, 255, 255)},
    "green": {"lower": (115, 185, 115), "upper": (170, 255, 185)},
    "red": {"lower": (0, 0, 180), "upper": (100, 100, 255)},
    "purple": {"lower": (140, 0, 140), "upper": (210, 75, 205)},
    "cyan": {"lower": (200, 185, 0), "upper": (255, 255, 65)},
}

feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)

if __name__ == "__main__":
    dir = "data/data_tagged"
    imgs = os.listdir(dir)
    for img in imgs[5:8]:
        try:
            hand = cv2.imread(os.path.join(dir, img), 1)
            print(hand.shape)
            for color, bounds in BGR_color_bounds.items():
                lowerb, upperb = bounds["lower"], bounds["upper"]
                mask = cv2.inRange(hand, lowerb=lowerb, upperb=upperb)
                _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
                hand_masked = cv2.bitwise_and(hand, hand, mask=mask)
                hand_masked = cv2.cvtColor(hand_masked, cv2.COLOR_BGR2GRAY)
                hand_masked =  cv2.threshold(hand_masked, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                hand_masked = hand_masked.astype(np.uint8)
                # print(mask.shape)
                # mask = mask.reshape(*mask.shape, 1)
                # corners = cv2.goodFeaturesToTrack(mask, **feature_params)
                # if corners is not None:
                #     for x, y in np.float32(corners).reshape(-1, 2):
                #         cv2.circle(a, (x, y), 10, (0, 255, 0), 1)

                print(hand_masked.shape)
                output = cv2.connectedComponentsWithStats(
                    hand_masked, 4, cv2.CV_8S  # connectivity  4 or 8
                )
                (num_labels, labels, stats, centroids) = output
                print(f"num labels: {num_labels}")
                print(f"labels: {labels}")
                print(f"stats: {stats}")
                print(f"centroids: {centroids}")
                
                plt.title(f"{img}-{color}")
                plt.imshow(hand_masked, cmap="gray")
                plt.show()
        except Exception as e:
            print(e)


