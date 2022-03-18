"""For each of these features, make sure there is a method in the `hand` class
called `get_<name of feature>`"""
ALL_FEATURE_NAMES = [
    "boneage",
    "gender",
    "gap_ratio_5",
    "gap_ratio_9",
    "gap_ratio_13",
    "gap_ratio_17",
    "gap_5",
    "gap_9",
    "gap_13",
    "gap_17",
    "gap_ratio_6",
    "gap_ratio_10",
    "gap_ratio_14",
    "gap_ratio_18",
    "carp_bones_max_distances",
    "carp_bones_max_distances_ratio",
    "carp_bones_max_diameter",
    "carp_bones_max_diameter_ratio",
    "max_purple_diameter",
    "max_purple_diameter_ratio",
    # "carp_bones_max_area",
    # "carp_bones_area_ratio",
    "epifisis_max_diameter",
    "epifisis_max_diameter_ratio",
    # "epifisis_max_area",
    "carp_bones_sum_perimeters",
    "yellow_sum_perimeters",
    "yellow_ratio_green",
    "carp_bones_sum_perimeters_ratio",
]

FEATURES_FOR_DATA_ANALYSIS = [
    "boneage",
    "gender",
    "gap_ratio_5",
    "gap_ratio_17",
    "gap_ratio_10",
    "gap_ratio_18",
    "gap_5",
    "gap_17",
    "carp_bones_max_distances",
    "carp_bones_max_diameter",
    "carp_bones_max_diameter_ratio",
    "max_purple_diameter_ratio",
    "epifisis_max_diameter",
    "epifisis_max_diameter_ratio",
    "carp_bones_sum_perimeters",
    "carp_bones_sum_perimeters_ratio",
    "yellow_sum_perimeters",
]


FEATURES_FOR_DATA_ANALYSIS = [
    "boneage",
    "gender",
    "gap_ratio_5",
    "gap_ratio_17",
    "gap_ratio_10",
    "gap_ratio_18",
    "gap_5",
    "gap_17",
    "carp_bones_max_distances",
    "carp_bones_max_diameter",
    "carp_bones_max_diameter_ratio",
    "max_purple_diameter_ratio",
    "epifisis_max_diameter",
    "epifisis_max_diameter_ratio",
    #"carp_bones_sum_perimeters",
    #"carp_bones_sum_perimeters_ratio",
    #"yellow_sum_perimeters",
]


AGE_BOUNDS = [
    (12 * (age_center - 1), 12 * (age_center + 1)) for age_center in range(3, 19)
]

AGE_BOUNDS=[(12*12, 16*12)]

hand_img_folder = "./data/boneage-training-dataset"
colored_data_dir = "./data/data_tagged"
# It is harder for mediapipe to recognize hand landmarks using the
# segmented hands (no idea why), so I am using both file versions
segmented_hand_img_folder = "./data/jsons"
hand_metadata_folder = "./data/boneage-training-dataset.csv"
batch_size = 1100
features_df_path = "./data/features_df.csv"
training_sample_size_ratio = 0.8
annotate_imgs = False
shuffle_data = True
allow_hand_plotting = True
do_affine_transform = False

import cv2

default_img = cv2.imread("data/data_tagged/1377.png", 0)
matrices_dir = "data/matrices"
affine_matrices_dir = matrices_dir
if do_affine_transform:
    hand_img_folder = "data/affines"
