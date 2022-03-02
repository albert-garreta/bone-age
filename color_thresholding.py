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



def process_connected_components(hand_masked):
    cc_output = cv2.connectedComponentsWithStats(
        hand_masked, 4, cv2.CV_8S  # connectivity  4 or 8
    )

    (num_labels, labels, stats, centroids) = cc_output
    print(f"num labels: {num_labels}")
    areas = [stats[label, cv2.CC_STAT_AREA] for label in range(2,num_labels)]
    print(areas)
    total_area = sum(areas)
    area_ratios = [round(area/total_area,4) for area in areas]
    print(area_ratios)
    for label in range(2,num_labels): # the 0 and 2 are exterior and bounding box
        print(f"Examining label: {label}")
        starting_x_coord = stats[label, cv2.CC_STAT_LEFT]
        starting_y_coord = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[label]
        output = hand_masked.copy()
        print(f"Component area", area)
        cv2.circle(output, (int(cX), int(cY)), 8, (0, 0, 255), -1)
        cv2.rectangle(output, (starting_x_coord, starting_y_coord), (starting_x_coord + width, starting_y_coord + height), (0, 0, 255, 0), 3)
        plt.imshow(output)
        plt.show()

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
                process_connected_components(hand_masked)
        except Exception as e:
            print(e)
