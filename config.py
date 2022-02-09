FEATURE_NAMES = [
    "boneage",
    "gender",
    "ratio_finger_palm",
    "gap_proxy_mean",
    "gap_proxy_std",
    #"gap_proxy_skew",
    "ratio_finger_to_gap_std",
    "ratio_finger_to_gap_mean",
    #"ratio_finger_to_gap_skew",
]
hand_img_folder = "./data/boneage-training-dataset"
hand_metadata_folder = "./data/boneage-training-dataset.csv"
batch_size = 1000
features_df_path = './data/features_df.csv'
training_sample_size = 800
annotate_imgs = True
shuffle_data = True