"""
Draws scatterplots of all features in config.FEATURES_FOR_DATA_ANALYSIS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import config

features = [x for x in config.FEATURES_FOR_DATA_ANALYSIS if x not in ["boneage", "gender"]]

d = pd.read_csv("./data/features_df.csv")
d = d.loc[d["gender"]==1,:]
boneages = d["boneage"]
boneages = [round(x/12,4) for x in list(d["boneage"])]
for feature in features:
    feat_values = d[feature]
    sbn.scatterplot(boneages, feat_values)
    plt.title(feature)
    plt.xticks(np.arange(0, 20, 1))
    plt.show()
