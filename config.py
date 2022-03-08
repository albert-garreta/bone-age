"""For each of these features, make sure there is a method in the `hand` class
called `get_<name of feature>`"""
ALL_FEATURE_NAMES = [
    "boneage",
    "gender",
    "metacarp_20_23_gap",
    "metacarp_27_30_gap",
    "metacarp_4_7_gap",
    "metacarp_12_15_gap",
    "carp_bones_max_diameter",
    "epifisis_diameter",
]

FEATURES_FOR_DATA_ANALYSIS = [
    "boneage",
    "gender",
    "metacarp_gap",
    "carp_bones_area_ratio",
    "epifisis_area_ratio",
]

hand_img_folder = "./data/boneage-training-dataset"
# It is harder for mediapipe to recognize hand landmarks using the
# segmented hands (no idea why), so I am using both file versions
segmented_hand_img_folder = "./data/data_tagged"
hand_metadata_folder = "./data/boneage-training-dataset.csv"
batch_size = 500
features_df_path = "./data/features_df.csv"
training_sample_size = int(0.75 * batch_size)
annotate_imgs = False
shuffle_data = True
allow_hand_plotting = False
