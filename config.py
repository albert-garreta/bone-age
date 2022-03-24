"""For each of these features, make sure there is a method in the `hand` class
called `get_<name of feature>`"""
from base64 import standard_b64decode


ALL_FEATURE_NAMES = [
    "id",
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
    "epifisis_max_diameter_ratio",
    "carp_bones_max_diameter_ratio",
    "gap_ratio_17",
    "gap_ratio_18",
    "gap_ratio_10",
    "gap_ratio_5",
    "carp_bones_max_diameter",
    "gap_5",
    "gap_17",
    "carp_bones_max_distances",
    "epifisis_max_diameter",
    "max_purple_diameter",
 "carp_bones_sum_perimeters",
 "carp_bones_sum_perimeters_ratio",
 "yellow_sum_perimeters",
]


FEATURES_FOR_DATA_ANALYSIS = [
    "boneage",
    "gender",
    "max_purple_diameter", # log
   # "epifisis_max_diameter_ratio", # log, but some older people have 0 epifisis-> bad
    "carp_bones_max_diameter_ratio", # log (some older boneages are 0 in this feat)
    "gap_ratio_17",
    "gap_ratio_5",
#     "gap_ratio_18",
#     "gap_ratio_10",
#     "carp_bones_max_diameter",
#     "gap_5",
#     "gap_17",
#     "carp_bones_max_distances",
#     "epifisis_max_diameter",
#  "carp_bones_sum_perimeters",
#  "carp_bones_sum_perimeters_ratio",
#  "yellow_sum_perimeters",
]


LOG_FEATS = ["max_purple_diameter", "carp_bones_max_diameter_ratio"]



#FEATURES_FOR_DATA_ANALYSIS = [
#    "boneage",
#    "gender",
#    #"gap_5",
#   
#]

AGE_BOUNDS = [
    (int(12 * (age_center - 4)), int(12 * (age_center + 4)))
    for age_center in [4, 8, 12, 16]
] + [(0, 20 * 12)]

# AGE_BOUNDS=[(12*12, 16*12)]

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
max_std_in_losses = 20
quartile_remove_outliers =1
standardize = True
import cv2

# default_img = cv2.imread("data/data_tagged/1377.png", 0)
matrices_dir = "data/matrices"
affine_matrices_dir = matrices_dir
if do_affine_transform:
    hand_img_folder = "data/affines"


FORBIDDEN_IMGS = [
    1451,
    1525,
    1578,
    1753,
    1972,
    2298,
    3681,
    4217,
    4921,
    10443,
    15070,
    1377,
    1388,
    1430,
    1452,
    1512,
    1516,
    1517,
    1553,
    1571,
    1577,
    1578,
    1591,
    1647,
    1671,
    1688,
    1714,
    1716,
    1738,
    1742,
    1753,
    1761,
    1810,
    1815,
    1821,
    1823,
    1868,
    1916,
    1941,
    1954,
    1955,
    1987,
    1988,
    1992,
    2002,
    2077,
    2291,
    2461,
    2531,
    2657,
    2862,
    3189,
    3235,
    3260,
    3294,
    3378,
    3534,
    3569,
    3589,
    3596,
    3622,
    3638,
    3722,
    3752,
    3758,
    3826,
    3847,
    4050,
    4064,
    4080,
    4085,
    4089,
    4108,
    4128,
    4137,
    4147,
    4160,
    4162,
    4168,
    4180,
    4193,
    4210,
    4219,
    4230,
    4235,
    4280,
    4284,
    4322,
    4575,
    4610,
    4942,
    5002,
    5149,
    5164,
    5228,
    5279,
    5379,
    5390,
    5479,
    5573,
    5831,
    5844,
    5927,
    6071,
    6213,
    6351,
    6504,
    6796,
    6841,
    6848,
    6850,
    7049,
    7511,
    7619,
    7627,
    7671,
    7988,
    8074,
    8083,
    8186,
    8295,
    8254,
    8709,
    8910,
    8928,
    9057,
    9122,
    9909,
    10150,
    10315,
    10697,
    10739,
    11095,
    11103,
    11266,
    11375,
    11388,
    11705,
    12808,
    12894,
    12931,
    13287,
    13474,
    13999,
    14206,
    14822,
    14930,
]
#FORBIDDEN_IMGS = []
