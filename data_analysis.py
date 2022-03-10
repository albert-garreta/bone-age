import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle
import numpy as np
import config as config
from lib.hand_factory import HandFactory
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def main():
    #data_processor = HandFactory()
    #featurizer = Featurizer()
    #data_processor.get_features_dataframe()

    df = pd.read_csv(config.features_df_path).iloc[
        :, 1:
    ]  # remove Unnamed: 0 -- how to abvoid having it in the first place?
    df = df[config.FEATURES_FOR_DATA_ANALYSIS]
    #df["carp_bones_area_ratio"] = np.sqrt(df["carp_bones_area_ratio"])

    if config.shuffle_data:
        # This shuffles the rows
        df = shuffle(df).reset_index(drop=True)
    
    # df[["gap_ratio_5",
    # "gap_ratio_9",
    # "gap_ratio_13",
    # "gap_ratio_17",]] = np.log(1+df[["gap_ratio_5",
    # "gap_ratio_9",
    # "gap_ratio_13",
    # "gap_ratio_17",]])
    
    
    df = df.dropna()

    df = df.loc[df["gender"] ==1, :]  # only male/female
    
    
    #
    
    #print(df)
    #df = df.loc[df["carp_bones_area_ratio"] > 0, :] 
    #df = df.loc[df["carp_bones_area_ratio"] <1, :] 
    #print(df)
    
    training_data_size = int(len(df)*config.training_sample_size_ratio)
    
    #df["boneage"] = np.log(df["boneage"])
    
    print(df.corr())

    
    df_train = df.iloc[: training_data_size]
    df_test = df.iloc[-training_data_size :]
    df_test.index = range(len(df_test))
    
    
    
    target_train = df_train["boneage"]
    target_test = df_test["boneage"]
    df_train = df_train.drop("boneage", axis=1)
    df_test = df_test.drop("boneage", axis=1)
    
    PN = PolynomialFeatures(2)
    #df_train = PN.fit_transform(df_train)
    #df_test = PN.fit_transform(df_test)
    
    reg = linear_model.LinearRegression()
    scaler  =StandardScaler()
    
    df_train = scaler.fit_transform(df_train)
    reg = reg.fit(df_train, target_train)
    
    df_test = scaler.transform(df_test)
    preds = reg.predict(df_test)
    loss = abs(preds - target_test)
    print(loss.mean())
    print(reg.coef_)
    print(reg.intercept_)
    # for i in range(len(preds)):
    #    print(target_test.iloc[i], preds[i])


if __name__ == "__main__":
    main()
