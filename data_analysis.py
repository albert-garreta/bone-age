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
from datetime import datetime

def remove_forbidden_imgs(df):
    #TODO: improve?
    for forb_img in config.FORBIDDEN_IMGS:
        df = df.loc[df["id"]!=forb_img]
    return df
def main_train_test_fun(df):

    df = shuffle(df).reset_index(drop=True)
    training_data_size = int(len(df) * config.training_sample_size_ratio)

    df = df[config.FEATURES_FOR_DATA_ANALYSIS]

    df_train = df.iloc[:training_data_size]
    df_test = df.iloc[training_data_size:]
    if len(df_train) == 0 or len(df_test) == 0:
        return None
    df_test.index = range(len(df_test))

    target_train = df_train["boneage"]
    target_test = df_test["boneage"]
    df_train = df_train.drop("boneage", axis=1)
    df_test = df_test.drop("boneage", axis=1)

    reg = linear_model.LinearRegression()
    scaler = StandardScaler()

    df_train = scaler.fit_transform(df_train)
    reg = reg.fit(df_train, target_train)

    df_test = scaler.transform(df_test)
    preds = reg.predict(df_test)
    disparity = preds - target_test
    return disparity


def process_data_by_gender_and_age_bounds(df, gender, age_bounds):
    try:
        df_restricted = df.loc[df["gender"] == gender, :]  # only male/female
        df_restricted = df_restricted.loc[df_restricted["boneage"] <= age_bounds[1]]
        df_restricted = df_restricted.loc[df_restricted["boneage"] >= age_bounds[0]]
        training_data_size = int(len(df_restricted) * config.training_sample_size_ratio)
        test_data_size = len(df_restricted) - training_data_size
        print("Size training and test data:", training_data_size, test_data_size)
        return df_restricted
    except Exception as e:
        #raise Exception
        print(e)
        return None



def main():
    df = pd.read_csv(config.features_df_path).iloc[
        :, 1:
    ]  # remove Unnamed: 0 -- how to abvoid having it in the first place?
    print(df)
    df = remove_forbidden_imgs(df)
    df = df.drop(columns=["id"])
    df = df[config.FEATURES_FOR_DATA_ANALYSIS]
    print(df.corr())
    print(df.head())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    
    with open("./logs/test.log", "w") as f:
        f.write(f"Test results for {datetime.now()}\n\n")
    
    all_losses_std = []
    all_losses_mean = []
    all_num_samples_for_range = []
    for age_bounds in config.AGE_BOUNDS:
        losses_std = []
        losses_mean = []
        num_samples_for_range = 0
        for gender in [0,1]:
            print(age_bounds, gender)
            df_restricted = process_data_by_gender_and_age_bounds(df, gender, age_bounds)
            num_samples_for_range += len(df_restricted)
            for _ in tqdm(range(100)):
                disparity = main_train_test_fun(df_restricted)
                loss = abs(disparity)
                losses_std.append(np.std(loss))
                losses_mean.append(np.mean(loss))
        msg = f"Age bounds {age_bounds[0]/12} <= age (years) <= {age_bounds[1]/12}\n    Std of age disparity: {round(np.mean(losses_std),3)} (months). Mean absolute error: {round(np.mean(losses_mean),3)}. Num samples for range: {num_samples_for_range}\n"
        # with open("./logs/test.log", "a") as f:
        #     f.write(msg)
        all_losses_mean.append(round(np.mean(losses_mean),3))
        all_losses_std.append(round(np.mean(losses_std),3))
        all_num_samples_for_range.append(num_samples_for_range)
    age_bounds_in_years = [(x[0]/12, x[1]/12) for x in config.AGE_BOUNDS]
    df_results = pd.DataFrame({"age_bounds (years)": age_bounds_in_years, "Std age disparity (months)": all_losses_std, "Mean error": all_losses_mean, "Num samples": all_num_samples_for_range})
    print(df_results)
    df_results.to_csv("./logs/test_results.csv")  
    msg = f"{df_results}"
    with open("./logs/test.log", "a") as f:
        f.write(msg)
    
if __name__ == "__main__":
    main()
