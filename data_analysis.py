import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import config as config
from lib.hand_factory import HandFactory
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as st


def main_train_test_fun(df):
    df = shuffle(df).reset_index(drop=True)
    training_data_size = int(len(df) * config.training_sample_size_ratio)
    
    df = df[config.FEATURES_FOR_DATA_ANALYSIS]
    
    df_train = df.iloc[:training_data_size]
    df_test = df.iloc[training_data_size:]
    df_test.index = range(len(df_test))

    target_train = df_train["boneage"]
    target_test = df_test["boneage"]
    df_train = df_train.drop("boneage", axis=1)
    df_test = df_test.drop("boneage", axis=1)

    PN = PolynomialFeatures(2)
    # df_train = PN.fit_transform(df_train)
    # df_test = PN.fit_transform(df_test)

    reg = linear_model.LinearRegression()
    scaler = StandardScaler()

    df_train = scaler.fit_transform(df_train)
    reg = reg.fit(df_train, target_train)

    df_test = scaler.transform(df_test)
    preds = reg.predict(df_test)
   # print(preds)
    #print(target_test)
    disparity = preds - target_test
    return disparity


def main():
    df = pd.read_csv(config.features_df_path).iloc[
        :, 1:
    ]  # remove Unnamed: 0 -- how to abvoid having it in the first place?
    df = df[config.FEATURES_FOR_DATA_ANALYSIS]
    print(df.corr())
    print(df.head())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    for gender in [0,1]:
        for age_center in range(19, 20):#[12*year for year in range(0,20)]:
            df_restricted = df.loc[df["gender"] == gender, :]  # only male/female
            #df_restricted = df_restricted.loc[df_restricted["boneage"] <= 12*(age_center + 1),:]
            #df_restricted = df_restricted.loc[df_restricted["boneage"] >=  12*(age_center - 1),:]
            training_data_size = int(len(df_restricted) * config.training_sample_size_ratio)
            test_data_size = len(df_restricted) -training_data_size
            print("Size training and test data:", training_data_size, test_data_size)
            
            losses = []
            if training_data_size >0 and test_data_size>0:
                for _ in tqdm(range(1)):
                    disparity = main_train_test_fun(df_restricted)
                    print(disparity)
                    loss = abs(disparity)
                    losses.append(loss.mean())
                    interval = st.t.interval(alpha=0.95, df=len(disparity)-1, loc=np.mean(disparity), scale=st.sem(disparity)) 
                    print(interval)
                    print(np.std(loss))
                    pd.Series(disparity).hist()
                    plt.show()
                print(f"Average losses for gender {gender} and age center {age_center}: ", np.mean(losses))
            

    # print(reg.coef_)
    # print(reg.intercept_)


if __name__ == "__main__":
    main()
