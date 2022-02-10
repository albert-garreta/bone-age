import pandas as pd
import config
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
df = pd.read_csv(config.features_df_path).iloc[:, 1:] # remove Unnamed: 0 -- how to abvoid having it in the first place?

df = df[config.FEATURES_FOR_DATA_ANALYSIS]

if config.shuffle_data:
    # This shuffles the rows
    df = shuffle(df).reset_index(drop=True)
    
print(df.corr())

df = df.dropna()


df = df.loc[df['gender']==1,:] # only male/female

df_train = df.iloc[: config.training_sample_size]
df_test = df.iloc[-config.training_sample_size :]
df_test.index = range(len(df_test))
target_train = df_train["boneage"]
target_test = df_test["boneage"]
df_train = df_train.drop("boneage", axis=1)
df_test = df_test.drop("boneage", axis=1)


reg = linear_model.LinearRegression()
reg = reg.fit(df_train, target_train)


preds = reg.predict(df_test)

loss = abs(preds - target_test)
print(loss.mean())
#print(np.sqrt(loss.mean()))
print(reg.coef_)

#for i in range(len(preds)):
#    print(target_test.iloc[i], preds[i])
