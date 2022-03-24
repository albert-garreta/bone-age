from dis import dis
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
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
    # TODO: improve efficiency
    for forb_img in config.FORBIDDEN_IMGS:
        df = df.loc[df["id"] != forb_img]
    return df


def remove_outlier_samples(df):
    # TODO: do it without a for?
    for feature in [
        x for x in config.FEATURES_FOR_DATA_ANALYSIS if x not in ["boneage", "gender"]
    ]:
        # print(df[feature].quantile(0.95))
        df = df[df[feature] < df[feature].quantile(config.quartile_remove_outliers)]
        # print(df)
        # assert False
    return df


def main_train_test_fun(df):

    df = shuffle(df).reset_index(drop=True)
    df = remove_outlier_samples(df)

    training_data_size = int(len(df) * config.training_sample_size_ratio)

    df = df[config.FEATURES_FOR_DATA_ANALYSIS]
    
    #df["boneage"] = np.log(df["boneage"])
    
    df = df.loc[df["max_purple_diameter"]>0]
    df = df.loc[df["gap_ratio_5"]<0.05]
    df = df.loc[df["carp_bones_max_diameter_ratio"]>0]
        
    df_train = df.iloc[:training_data_size]
    df_test = df.iloc[training_data_size:]
    if len(df_train) == 0 or len(df_test) == 0:
        return None, None
    df_test.index = range(len(df_test))
    
    target_train = df_train["boneage"]
    target_test = df_test["boneage"]
    df_train = df_train.drop("boneage", axis=1)
    df_test = df_test.drop("boneage", axis=1)
   # df_train = df_train.drop("gender", axis=1)
   # df_test = df_test.drop("gender", axis=1)
    
    #print(df_train.head())
    final_train_columns = df_train.columns
    reg = linear_model.LinearRegression()
    scaler = StandardScaler()
    
    if config.standardize:
        df_train = scaler.fit_transform(df_train)
    reg = reg.fit(df_train, target_train)
    
    feature_scores = SelectKBest(score_func=f_regression, k='all')
    #feature_scores.fit(df_train, target_train)
    

    if config.standardize:
        df_test = scaler.transform(df_test)
    preds = reg.predict(df_test)
    #preds = np.exp(preds)
    #target_test = np.exp(target_test)
    #print(reg.coef_)
    disparity = preds - target_test
    return disparity, None#, feature_scores.scores_


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
        # raise Exception
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
    all_max_errors = []
    for age_bounds in config.AGE_BOUNDS:
        losses_std = []
        losses_mean = []
        max_errors = []
        num_samples_for_range = 0
        for gender in [0, 1]:
            print(age_bounds, gender)
            df_restricted = process_data_by_gender_and_age_bounds(
                df, gender, age_bounds
            )
            # print(df_restricted)
            num_samples_for_range += len(df_restricted)
            if num_samples_for_range > 10:
                for _ in tqdm(range(100)):
                    disparity, feat_scores = main_train_test_fun(df_restricted)
                    disparity = pd.Series(disparity)
                    disparity = disparity[disparity< np.mean(disparity) + config.max_std_in_losses*np.std(disparity)]
                    loss = pd.Series(abs(disparity))
                    losses_std.append(np.std(loss))
                    losses_mean.append(np.mean(loss))
                    max_errors.append(np.max(disparity))
                    #pd.Series(disparity).hist()
                    #plt.show()
            #for idx, col in enumerate([x for x in config.FEATURES_FOR_DATA_ANALYSIS if x not in ["boneage", "gender"]]):
                #print(col,feat_scores[idx])

        msg = f"Age bounds {age_bounds[0]/12} <= age (years) <= {age_bounds[1]/12}\n    Std of age disparity: {round(np.mean(losses_std),3)} (months). Mean absolute error: {round(np.mean(losses_mean),3)}. Num samples for range: {num_samples_for_range}\n"
        # with open("./logs/test.log", "a") as f:
        #     f.write(msg)
        all_losses_mean.append(round(np.mean(losses_mean), 3))
        all_losses_std.append(round(np.mean(losses_std), 3))
        all_num_samples_for_range.append(num_samples_for_range)
        all_max_errors.append(np.round(np.mean(max_errors),3))

    age_bounds_in_years = [(x[0] / 12, x[1] / 12) for x in config.AGE_BOUNDS]
    df_results = pd.DataFrame(
        {
            "age_bounds (years)": age_bounds_in_years,
            "Std age disparity (months)": all_losses_std,
            "Mean error": all_losses_mean,
            "Max disparity": all_max_errors,
            "Num samples": all_num_samples_for_range,
        }
    )
    print(df_results)
    df_results.to_csv("./logs/test_results.csv")
    msg = f"{df_results}"
    with open("./logs/test.log", "a") as f:
        f.write(msg)


if __name__ == "__main__":
    main()
