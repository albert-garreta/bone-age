from sys import hash_info
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

HSV_color_bounds = {
    "yellow": {"lower": (55, 1, 0), "upper": (65, 255, 255)},
    "green": {"lower": (115, 1, 0), "upper": (125, 255, 255)},
    "red": {"lower": (250, 1, 0), "upper": (5, 255, 255)},
    "purple": {"lower": (295, 1, 0), "upper": (305, 255, 255)},
    "cyan": {"lower": (178, 1, 0), "upper": (188, 255, 255)},
}

def from_hsv_to_bgr(color_value):
    return cv2.cvtColor(np.uint8([[list(color_value)]]), cv2.COLOR_HSV2BGR)[0][0]
    
BGR_color_bounds = {
    key:  {"lower": from_hsv_to_bgr(value["lower"]), "upper": from_hsv_to_bgr(value["upper"])} for key,value in HSV_color_bounds.items()
}
print(BGR_color_bounds)

connected_comp_valid_area_bounds = {
    "yellow": (0, np.inf),
    "green": (2000, np.inf),
    "red": (),
}


feature_params = dict(maxCorners=500, qualityLevel=0.2, minDistance=15, blockSize=9)


def show(img):
    plt.figure(figsize=[18, 10])
    plt.imshow(img, cmap="gray")
    plt.show()


def connected_comp_is_valid(width, height, widths, heights):
    return width / sum(widths) > 0.001 and height / sum(heights) > 0.001


def process_connected_components(hand_masked):

    cc_output = cv2.connectedComponentsWithStats(
        hand_masked, 8, cv2.CV_8S  # connectivity  4 or 8
    )

    (num_labels, labels, stats, centroids) = cc_output
    print(f"num labels: {num_labels}")
    areas = [stats[label, cv2.CC_STAT_AREA] for label in range(2, num_labels)]
    print(areas)
    total_area = sum(areas)
    area_ratios = [round(area / total_area, 4) for area in areas]
    print(area_ratios)

    widths = [stats[label, cv2.CC_STAT_WIDTH] for label in range(2, num_labels)]
    heights = [stats[label, cv2.CC_STAT_HEIGHT] for label in range(2, num_labels)]
    total_widths = sum(widths)
    widths_ratios = [round(w / total_widths, 4) for w in widths]
    print("widths_ratios", widths_ratios)
    total_heights = sum(heights)
    heights_ratios = [round(area / total_heights, 4) for area in heights]
    print("heights_ratios", heights_ratios)

    for label in range(0, num_labels):  # the 0 and 2 are exterior and bounding box
        print(f"Examining label: {label}")
        starting_x_coord = stats[label, cv2.CC_STAT_LEFT]
        starting_y_coord = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[label]

        if connected_comp_is_valid(width, height, widths, heights):
            output = hand_masked.copy()
            print(f"Component area", area)
            cv2.circle(output, (int(cX), int(cY)), 8, (0, 0, 255), -1)
            cv2.rectangle(
                output,
                (starting_x_coord, starting_y_coord),
                (starting_x_coord + width, starting_y_coord + height),
                (0, 0, 255, 0),
                3,
            )
            show(output)


def remove_gray(img_hsv):
    mask = cv2.inRange(img_hsv, (0, 1, 0), (255, 255, 255))
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    show(img_bgr)
    img_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    show(img_bgr)
    #img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    #show(img_hsv)
    return img_bgr

def remove_gray(img_hsv):
    mask = cv2.inRange(img_hsv, (0, 1, 0), (255, 255, 255))
    img_bgr = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    show(img_hsv)
    #img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    #show(img_hsv)
    return img_hsv

def apply_max_saturation_and_value(img_hsv):
    h,s,v = img_hsv.split()
    s_new = 255*np.ones(s.shape)
    v_new = 255*np.ones(v.shape)
    return cv2.merge(h, s_new, v_new)

if __name__ == "__main__":
    dir = "data/data_tagged"
    imgs = os.listdir(dir)
    for img in imgs[6:15]:
        try:
            hand = cv2.imread(os.path.join(dir, img), 1)
            hand = cv2.cvtColor(hand, cv2.COLOR_BGR2RGB)
            hand_hsv = cv2.cvtColor(hand, cv2.COLOR_RGB2HSV)
            hand_hsv = remove_gray(hand_hsv)
            #hand_hsv = apply_max_saturation_and_value(hand_hsv)
            #print(hand_hsv.shape)
            # hand_masked = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
            # show(hand_masked)
            # _, hand_masked = cv2.threshold(hand_masked, 10, 255, cv2.THRESH_BINARY)
            # hand_masked = cv2.adaptiveThreshold(hand_masked, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
            # _, hand_masked = cv2.threshold(hand_masked, 100, 255, cv2.THRESH_BINARY)
            # show(hand_masked)
            # contours, hierarchy = cv2.findContours(hand_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print(len(contours))
            # hand_masked = cv2.cvtColor(hand_masked, cv2.COLOR_GRAY2BGR)
            # cv2.drawContours(hand_masked, contours, -1, (0,255,0), 3)
            # show(hand_masked)
            # hand = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
            # hand = hand.astype(np.uint8)
            # hand =  cv2.threshold(hand, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            # process_connected_components(hand)
            # hand_masked = cv2.threshold(
            #     hand_masked, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
            # )[1]
            # hand_masked = hand_masked.astype(np.uint8)
            # show(hand_masked)

            # process_connected_components(hand_masked)

            for color, bounds in HSV_color_bounds.items():
                #show(hand_hsv)
                lowerb, upperb = bounds["lower"], bounds["upper"]
                hand_bgr_copy = hand_hsv.copy()
                mask = cv2.inRange(hand_bgr_copy, lowerb=lowerb, upperb=upperb)
                show(mask)
                # _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
                # hand_bgr_copy = cv2.cvtColor(hand_bgr_copy, cv2.COLOR_HSV2BGR)
                hand_gray = cv2.cvtColor(hand_bgr_copy, cv2.COLOR_BGR2GRAY)
                hand_gray = cv2.bitwise_and(hand_gray, hand_gray, mask=mask)
                hand_gray = cv2.threshold(
                    hand_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
                )[1]
                hand_gray = hand_gray.astype(np.uint8)
                show(hand_gray)
                #
                process_connected_components(hand_gray)
        except Exception as e:
            print(e)
