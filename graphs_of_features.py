import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np
import config

features = [x for x in config.FEATURES_FOR_DATA_ANALYSIS if x not in ["boneage", "gender"]]

d = pd.read_csv("./data/features_df.csv")
d = d.loc[d["gender"]==1,:]
boneages = d["boneage"]
#print(len(boneages))
boneages = [round(x/12,4) for x in list(d["boneage"])]
#print(len(boneages))
for feature in features:
    feat_values = d[feature]
    #feat_values = feat_values[feat_values < feat_values.quantile(1)]

    sbn.scatterplot(boneages, feat_values)
    plt.title(feature)
    plt.xticks(np.arange(0, 20, 1))
    #plt.yticks(np.arange(0, max(y), 2))
    plt.show()
    
#for feature in config.LOG_FEATS:
#    feat_values = np.exp(d[feature])
#    feat_values = feat_values[feat_values < feat_values.quantile(0.8)]
#
#    sbn.scatterplot(boneages, feat_values)
#    plt.title(feature)
#    plt.show()