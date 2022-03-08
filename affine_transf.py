import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_transformation(default_lungs, segm_ret):

    # Search for the best afin transformation which applied to default_lungs gets segm_ret. Returns default_lungs with the transformation found.

    a = 1
    image_size = segm_ret.shape
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-7)
    try:
        (cc, warp_matrix) = cv2.findTransformECC(
            segm_ret, default_lungs, warp_matrix, cv2.MOTION_AFFINE, criteria
        )
        default_lungs = cv2.warpAffine(
            default_lungs,
            warp_matrix,
            (image_size[1], image_size[0]),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except:
        default_lungs = np.zeros((image_size[1], image_size[0]))
        a = 2

    return [default_lungs, a, warp_matrix]

if __name__ == '__main__':
    id1 = 10150
    id2 = 10214
    img1_dir= f"data/boneage-training-dataset/{id1}.png"
    img2_dir= f"data/boneage-training-dataset/{id2}.png"
    img1 = cv2.imread(img1_dir,0)
    img2 = cv2.imread(img2_dir,0)
    img1_new, a, matrix = find_transformation(img1, img2)
    print("a", a)
    print(matrix)
    plt.imshow(img1)
    plt.show()
    plt.imshow(img1_new)
    plt.show()
    plt.imshow(img2)
    plt.show()