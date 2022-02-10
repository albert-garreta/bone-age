ALL_FEATURE_NAMES = [
    "boneage",
    "gender",
    "ratio_finger_palm",
    # "gap_proxy_mean",
    # "gap_proxy_std",
    "ratio_finger_to_gap_std",
    "ratio_finger_to_gap_mean",
]
FEATURES_FOR_DATA_ANALYSIS = [
    "boneage",
    "gender",
    "ratio_finger_palm",
    #"gap_proxy_mean",
    #"gap_proxy_std",
    "ratio_finger_to_gap_std",
    "ratio_finger_to_gap_mean",
]
hand_img_folder = "./data/boneage-training-dataset"
hand_metadata_folder = "./data/boneage-training-dataset.csv"
batch_size = 10000
features_df_path = './data/features_df.csv'
training_sample_size = int(0.75*batch_size)
annotate_imgs = True
shuffle_data = True