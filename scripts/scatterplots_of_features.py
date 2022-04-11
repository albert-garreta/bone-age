import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import os
import sys

sys.path.append("../bone-age")
import config

"""
Draws scatterplots of all features in config.FEATURES_FOR_DATA_ANALYSIS for a given gender
The graphs are saved in bone-age/data/feature_scatterplots
"""

GENDER = 1

features = [
    x for x in config.FEATURES_FOR_DATA_ANALYSIS if x not in ["boneage", "gender", "id"]
]

if not os.path.exists("data/feature_scatterplots"):
    os.mkdir("data/feature_scatterplots/")

d = pd.read_csv("./data/features_df.csv")
# Remove to draw the scatterplot for all genders
d = d.loc[d["gender"] == 1, :]
boneages = d["boneage"]
# Write boneages in years (from months)
boneages = [round(x / 12, 4) for x in list(d["boneage"])]
for feature in features:
    feat_values = d[feature]
    sbn.scatterplot(boneages, feat_values)
    plt.title(feature)
    plt.xticks(np.arange(0, 20, 1))
    plt.savefig(f"data/feature_scatterplots/{feature}.png")
    plt.close() 