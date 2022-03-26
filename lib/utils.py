import os
import config


def remove_file_types_from_list(list_of_files):
    return [f.split('.')[0] for f in list_of_files]

def get_valid_hand_ids():
    files = os.listdir(config.hand_img_folder)
    files = remove_file_types_from_list(files)
    jsons = os.listdir(config.segmented_hand_img_folder)
    jsons = remove_file_types_from_list(jsons)
    valid_ids = [f for f in files if (f  in jsons and f not in config.FORBIDDEN_IMGS)]
    return valid_ids


def get_working_hand_img_files():
    valid_ids = get_valid_hand_ids()
    return [os.path.join(config.hand_img_folder, f"{id}.png") for id in valid_ids]

 