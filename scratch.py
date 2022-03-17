import cv2
import os
dir = "data/data_tagged"
import seaborn as sbn
import pandas as pd
from tqdm import tqdm
from lib.hand import Hand
import segmentations
from lib.hand_factory import get_id_from_file_name
import json

if __name__ == "__main__":
    for id in os.listdir("data/boneage-training-dataset"):
        try:
            hand_file = f"data/boneage-training-dataset/{id}"
            id =hand_file.split('/')[-1].split(".")[0]
            img = cv2.imread(hand_file, 1)
            segments = segmentations.get_segmentations(str(id))
            #print(segments)
            hand = Hand(img, 0, 0, id, segments)
            hand.get_hand_landmarks()
            hand.draw_landmarks()
            hand.img = segmentations.draw_all_contours(hand.img, hand.segments)
            hand.show()
            
            # hand.organize_segmentations()
            # ordered_segment_ids = hand.ordered_segment_ids
            # print(ordered_segment_ids)
            # hand = Hand(img, 0, 0, id, segments)
            # hand.img = segmentations.draw_all_contours(hand.img, hand.segments, order=ordered_segment_ids)
            # hand.show()
        except Exception as e:
            print(e)
        #hand.organize_segmentations()
        #success = hand.featurize()


# widths = []
# heights = []
# for im_dir in tqdm(os.listdir(dir)):
#     try:
#         img = cv2.imread(os.path.join(dir, im_dir),1)
#         #print(img.shape, img.shape[0]/img.shape[1])
#         widths.append(img.shape[0])
#         heights.append(img.shape[1])
#         
#     except: 
#         pass
# w = pd.Series(widths)
# print(w)
# w.hist(50)


    