import sys

sys.path.append("../bone-age")
import config
import pandas as pd
import os
import cv2
from tqdm import tqdm
from lib.hand import Hand
from lib import segmentations
import gc
from lib.utils import get_list_hand_files

# We create a dictionary where, for each possible feature creation error type, we save a list of all hand ids for which we
# obtain such type of error
error_types = ["read_file", "create_landmarks", "get_segmentaitons"] + [
    f"create_feature_{feat_name}" for feat_name in config.ALL_FEATURE_NAMES
]
error_info = {error_type: [] for error_type in error_types}


def get_features_dataframe() -> pd.DataFrame:
    """
    Creates self.features_df: a dataframe containing the features in FEATURES_FOR_DATA_ANALYSIS for each of the hands files
    in self.list_hand_files

    This df is saved in the folder specified by `config.features_df_path`
    """
    metadata_df = pd.read_csv(config.hand_metadata_folder)
    list_hand_files = get_list_hand_files()
    num_failed_files = 0

    # Here we will store, for each hand, a dictionary with each of the extracted features
    hand_features = []

    print("Extracting features from hands...")
    for hand_file in tqdm(list_hand_files):

        img_full_path = os.path.join(config.hand_img_folder, hand_file)
        img = cv2.imread(img_full_path)

        # hand_file has the format <ID>.png. Here we extract the <ID>
        id = int(hand_file.split(".")[0])

        # If cv2 wasn't able to read the file, we save `id` in the `error_info["hand_file"]` list
        # and we move on to the next hand
        if img is None:
            num_failed_files += 1
            error_info["read_file"].append(id)
            continue

        # Get the metadata (gender and boneage) corresponding to the specific hand under study
        hand_metadata = metadata_df.loc[metadata_df["id"] == id]
        age = int(hand_metadata["boneage"])
        gender = int(hand_metadata["male"].bool())

        # Here we attempt to get the segments corresponding to the hand (i.e. the bone contours we manually labelled)
        # `segments` is a dictionary stablishing the following map:
        # contour apearance order  --->  list of points forming the contour
        try:
            segments = segmentations.get_segmentations(str(id), img.shape)
        except Exception as e:
            print(e)
            error_info["get_segmentations"].append(id)
            num_failed_files += 1
            continue

        # Create an instance of the Hand class
        hand = Hand(img, age, gender, id, segments)

        # Attempts to get Google's hand landmarks
        success = hand.get_hand_landmarks()
        if not success:
            num_failed_files += 1
            error_info["create_landmarks"].append(id)
            continue

        # Attempts to create each and every one of the features listed in
        # `config.ALL_FEATURE_NAMES`.
        # If successful returns True, None, otherwise it returns False and the feature
        # name that first raised an Exception.
        # (This is maybe a little too dirty but let's not dwell here)
        success, feature_name_that_failed = hand.featurize()
        if success:
            if config.make_drawings:
                # It can be useful to draw stuff as features are being created, and then
                # save the resulting hand image
                cv2.imwrite(f"./data/drawings/{id}.png", hand.img)
            hand_features = get_feature_dict(hand)
        else:
            num_failed_files += 1
            error_info[f"create_feature_{feature_name_that_failed}"].append(id)
            continue

        # Garbage collect (without this the memory consumption was skyrocketing)
        gc.collect()

    # Save features into a DataFrame
    pd.DataFrame(hand_features).to_csv(config.features_df_path)

    # Process the error info
    print("Extraction complete")
    print(
        f"Failed to extract {num_failed_files} features out of {len(list_hand_files)} files"
    )
    print("Errors encountered in the following files:")
    with open("data/error_info.txt", "w") as f:
        for key, value in error_info.items():
            print(key, value)
            f.write(f"{key}: {value}")


def get_feature_dict(hand):
    return {feature: [getattr(hand, feature)] for feature in config.ALL_FEATURE_NAMES}


if __name__ == "__main__":
    get_features_dataframe()
