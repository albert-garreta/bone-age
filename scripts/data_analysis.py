from dis import dis
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.utils import shuffle
import numpy as np
import sys

sys.path.append("../bone-age")
import config as config
from lib.hand_factory import HandFactory
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.stats as st
from datetime import datetime
from sklearn.metrics import r2_score
import seaborn as sbn


def remove_forbidden_imgs(df):
    # TODO: improve efficiency
    for forb_img in config.FORBIDDEN_IMGS:
        df = df.loc[df["id"] != forb_img]
    return df


def cut_out_outlier_samples(df, feature, min=None, max=None):
    if feature not in df.columns:
        return df
    if min is not None:
        df = df.loc[df[feature] > min]
    if max is not None:
        df = df.loc[df[feature] < max]
    return df


def replace_outlier_samples(df, feature, min=None, max=None):
    if feature not in df.columns:
        return df
    if min is not None:
        df.loc[df[feature] > min, feature] = np.mean(df.loc[df[feature] > min, feature])
    if max is not None:
        df.loc[df[feature] < max, feature] = np.mean(df.loc[df[feature] < max, feature])
    return df


def remove_outlier_samples(df):
    # TODO: do it without a for?
    for feature in [
        x
        for x in config.FEATURES_FOR_DATA_ANALYSIS
        if x not in ["id", "boneage", "gender"]
    ]:
        # print(df[feature].quantile(0.95))
        df = df[df[feature] < df[feature].quantile(config.quartile_remove_outliers)]
        # print(df)
        # assert False
    return df


def prepare_and_split_train_test(df):

    df = shuffle(df).reset_index(drop=True)

    training_data_size = int(len(df) * config.training_sample_size_ratio)

    df_train = df.iloc[:training_data_size]
    df_test = df.iloc[training_data_size:]
    # print(df_train.shape, df_test.shape)
    if len(df_train) == 0 or len(df_test) == 0:
        return None, None
    df_test.index = range(len(df_test))

    target_train = df_train["boneage"]
    target_test = df_test["boneage"]
    #df_train = df_train.drop("boneage", axis=1)
    #df_test = df_test.drop("boneage", axis=1)
    # df_train = df_train.drop("gender", axis=1)
    # df_test = df_test.drop("gender", axis=1)
    return df_train, target_train, df_test, target_test


def main_train_test_fun(df_train, target_train, df_test, target_test):

    # print(df_train.head())
    
    df_train = df_train[config.FEATURES_FOR_DATA_ANALYSIS]
    df_test = df_test[config.FEATURES_FOR_DATA_ANALYSIS]

    final_train_columns = df_train.columns
    reg = linear_model.LinearRegression()
    scaler = StandardScaler()

    if config.standardize:
        df_train = scaler.fit_transform(df_train)
    reg = reg.fit(df_train, target_train)

    feature_scores = SelectKBest(score_func=f_regression, k="all")
    # feature_scores.fit(df_train, target_train)

    if config.standardize:
        df_test = scaler.transform(df_test)
    preds = reg.predict(df_test)
    # preds = np.exp(preds)
    # target_test = np.exp(target_test)
    # print(reg.coef_)
    disparity = preds - target_test
    
    train_preds = reg.predict(df_train)
    r2 = r2_score(target_train, train_preds)
    #r2 = r2_score(target_test, preds)
    return disparity, r2


def preprocess_data(df, gender, age_bounds):
    try:
        df_restricted = df.loc[df["gender"] == gender, :]  # only male/female
        df_restricted = df_restricted.loc[df_restricted["boneage"] <= age_bounds[1]]
        df_restricted = df_restricted.loc[df_restricted["boneage"] >= age_bounds[0]]
        training_data_size = int(len(df_restricted) * config.training_sample_size_ratio)
        test_data_size = len(df_restricted) - training_data_size
        # print("Size training and test data:", training_data_size, test_data_size)
        df = remove_outlier_samples(df)

        df = cut_out_outlier_samples(df, "max_purple_diameter", 0, 195)
        # df = cut_out_outlier_samples(df, "max_purple_diameter", None, 0.1)
        df = cut_out_outlier_samples(df, "epifisis_max_diameter_ratio", 0, None)
        # df = cut_out_outlier_samples(df, "epifisis_max_diameter_ratio", None, 0.001)

        df = cut_out_outlier_samples(df, "carp_bones_max_diameter_ratio", 0, None)
        df = cut_out_outlier_samples(df, "gap_ratio_5", None, 0.05)
        df = cut_out_outlier_samples(df, "gap_ratio_13", None, 0.13)
        df = cut_out_outlier_samples(df, "gap_ratio_9", None, 0.1)
        
        return df_restricted
    except Exception as e:
        # raise Exception
        print(e)
        return None
def get_worse_performing_samples(df, losses):
    df["losses"] = losses
    df.sort("losses")
    print( df["id"].iloc[-20:] )
    return df["id"].iloc[-20:]

def main(config):
    df = pd.read_csv(config.features_df_path).iloc[
        :, 1:
    ]  # remove Unnamed: 0 -- how to abvoid having it in the first place?
    df = remove_forbidden_imgs(df)
    # Avoid dropping the id to be able to study the worst-performing samples later
    # df = df.drop(columns=["id"]) 

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    with open("./logs/test.log", "w") as f:
        f.write(f"Test results for {datetime.now()}\n\n")

    all_losses_std = []
    all_losses_mean = []
    all_num_samples_for_range = []
    all_max_errors = []
    all_r2s = []
    for age_bounds in config.AGE_BOUNDS:
        losses_std = []
        losses_mean = []
        max_errors = []
        r2s = []
        num_samples_for_range = 0
        for gender in [0, 1]:
            # print(age_bounds, gender)
            df_restricted = preprocess_data(
                df, gender, age_bounds
            )
            # print(df_restricted)

            if num_samples_for_range > 10:
                for _ in range(500):
                    df_train, target_train, df_test, target_test = prepare_and_split_train_test(
                        df_restricted
                    )
                    
                    disparity, r2 = main_train_test_fun(
                        df_train, target_train, df_test, target_test
                    )
                    disparity = pd.Series(disparity)
                    disparity = disparity[
                        disparity
                        < np.mean(disparity)
                        + config.max_std_in_losses * np.std(disparity)
                    ]
                    loss = pd.Series(abs(disparity))
                    losses_std.append(np.std(loss))
                    losses_mean.append(np.mean(loss))
                    max_errors.append(np.max(disparity))
                    r2s.append(r2)
                    # pd.Series(disparity).hist()
                    # plt.show()
                num_samples_for_range += len(df_train)
                worst_performing_train_ids_last_iteration = get_worse_performing_samples(df_train, loss)
                predictions = np.array(disparity) + np.array(target_test)
            
            
            
        msg = f"Age bounds {age_bounds[0]/12} <= age (years) <= {age_bounds[1]/12}\n    Std of age disparity: {round(np.mean(losses_std),3)} (months). Mean absolute error: {round(np.mean(losses_mean),3)}. Num samples for range: {num_samples_for_range}\n"
        # with open("./logs/test.log", "a") as f:
        #     f.write(msg)
        all_losses_mean.append(round(np.mean(losses_mean), 3))
        all_losses_std.append(round(np.mean(losses_std), 3))
        all_num_samples_for_range.append(num_samples_for_range)
        all_max_errors.append(np.round(np.mean(max_errors), 3))
        all_r2s.append(np.mean(r2s))

    age_bounds_in_years = [(x[0] / 12, x[1] / 12) for x in config.AGE_BOUNDS]
    df_results = pd.DataFrame(
        {
            "age_bounds (years)": age_bounds_in_years,
            "Std age disparity (months)": all_losses_std,
            "Mean error": all_losses_mean,
            "Max disparity": all_max_errors,
            "R^2 score": all_r2s,
            "Num samples trained on": all_num_samples_for_range,
        }
    )
    print(df_results)
    df_results.to_csv("./logs/test_results.csv")
    msg = f"{df_results}"
    with open("./logs/test.log", "a") as f:
        f.write(msg)


def plot_predictions(true_labels, predictions):
    print(np.array(true_labels).reshape(len(true_labels), 1))
    print(predictions)
    sbn.scatterplot(true_labels, predictions)
    reg = linear_model.LinearRegression()
    true_labels = np.array(true_labels).reshape(len(true_labels), 1).astype(np.float)
    #predictions = np.array(predictions).reshape(len(predictions), 1)
    print(true_labels.shape)
    print(np.array(predictions).shape)
    print(true_labels)
    print(np.array(predictions))
    reg.fit(
        true_labels,
        predictions
    )

    true_labels = [month for month in range(12, 20 * 12, 10)]
    regressed_predictions = [reg.predict(np.array(true_label).reshape(1,-1)) for true_label in true_labels]

    print(regressed_predictions)

    sbn.lineplot(x=np.array(regressed_predictions).reshape(-1), y=np.array(regressed_predictions).reshape(-1))
    plt.show()


if __name__ == "__main__":
    print("ALL FEATURES")
    main(config)

    if True:

        for feat in [
            x for x in config.ALL_FEATURE_NAMES if x not in ["id", "boneage", "gender"]
        ]:
            print(f"\n\nFEATURE {feat}\n")
            config.FEATURES_FOR_DATA_ANALYSIS = ["boneage", "gender", feat]
            main(config)
