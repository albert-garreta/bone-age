import sys

# Permet importar codi de fora de la carpeta actual
sys.path.append("../bone-age")
# Arxiu de configuració del repositori
import config as config
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


"""Durant tot l'script, `df` es un DataFrame que conté les columnes `config.FEATURES_FOR_DATA_ANALYSIS`
"""


def remove_forbidden_imgs(df):
    # Treu les imatges del df que han estat "prohibides" perquè el model de google no troba els landmarks
    # correctament (llista obtinguda fent inspecció a ull)
    # TODO: improve efficiency
    for forb_img in config.FORBIDDEN_IMGS:
        df = df.loc[df["id"] != forb_img]
    return df


def cut_out_outlier_samples(
    df: pd.DataFrame, feature: str, min=-np.inf, max=np.inf
) -> pd.DataFrame:
    """Removes all rows from df for which the feature `feature` has value below min or above max,
    assuming min!=None and max != None, respectively
    """
    if min is not None:
        df = df.loc[df[feature] > min]
    if max is not None:
        df = df.loc[df[feature] < max]
    return df


def remove_outlier_samples_by_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """For each feature, removes all rows from `df` whose feature value
    is above `config.quantile_remove_outliers`"""

    # TODO: do it without a for?
    for feature in [
        x
        for x in config.FEATURES_FOR_DATA_ANALYSIS
        if x not in ["id", "boneage", "gender"]
    ]:
        df = df[df[feature] < df[feature].quantile(config.quartile_remove_outliers)]
    return df


def remove_outlier_samples_by_bounds(
    df: pd.DataFrame, feature: str, lower_bound, upper_bound
) -> pd.DataFrame:
    """Removes all rows from df for which the feature `feature` has value below lower_bound or above upper_bound,
    assuming lower_bound!=None and upper_bound != None, respectively
    """
    if lower_bound is not None:
        df = df.loc[df[feature] > lower_bound]
    if upper_bound is not None:
        df = df.loc[df[feature] < upper_bound]
    return df


def prepare_and_split_train_test(df):

    df = remove_outlier_samples_by_quartile(df)

    # The following bounds have been determined by inspecting the scatterplots of
    # each feature
    df = remove_outlier_samples_by_bounds(df, "max_purple_diameter", 0, 195)
    # df = cut_out_outlier_samples(df, "max_purple_diameter", None, 0.1)
    df = remove_outlier_samples_by_bounds(df, "epifisis_max_diameter_ratio", 0, None)

    # df = cut_out_outlier_samples(df, "epifisis_max_diameter_ratio", None, 0.001)
    df = remove_outlier_samples_by_bounds(df, "carp_bones_max_diameter_ratio", 0, None)
    df = remove_outlier_samples_by_bounds(df, "gap_ratio_5", None, 0.05)
    df = remove_outlier_samples_by_bounds(df, "gap_ratio_13", None, 0.13)
    df = remove_outlier_samples_by_bounds(df, "gap_ratio_9", None, 0.1)

    # Shuffling!
    df = shuffle(df).reset_index(drop=True)
    df = df[config.FEATURES_FOR_DATA_ANALYSIS]

    # Split into training and test sets, and separate the target variable
    training_data_size = int(len(df) * config.training_sample_size_ratio)
    df_train = df.iloc[:training_data_size]
    df_test = df.iloc[training_data_size:]
    if len(df_train) == 0 or len(df_test) == 0:
        return None, None, None, None
    df_test.index = range(len(df_test))
    target_train = df_train["boneage"]
    target_test = df_test["boneage"]
    df_train = df_train.drop("boneage", axis=1)
    df_test = df_test.drop("boneage", axis=1)
    return df_train, target_train, df_test, target_test


def main_train_test_fun(df_train, target_train, df_test, target_test, metrics):
    """Applies a simple linear regression"""

    reg = linear_model.LinearRegression()
    scaler = StandardScaler()

    df_train = df_train.drop(columns="id")
    df_test = df_test.drop(columns="id")

    # No necessitem standarditzar de fet, ja que els parámetres d'standardització
    # els "absorbeix" la regressó
    if config.standardize:
        df_train = scaler.fit_transform(df_train)
    reg = reg.fit(df_train, target_train)

    if config.standardize:
        df_test = scaler.transform(df_test)
    preds = reg.predict(df_test)
    difference = preds - target_test

    # Registrem el R2 sobre el training dataset
    train_preds = reg.predict(df_train)
    r2 = r2_score(target_train, train_preds)

    loss = pd.Series(abs(difference))
    # !! Note np.std(loss) is the same as np.std(differences), which is the
    # metric we care most about
    metrics.std_losses.append(np.std(loss))
    metrics.mean_losses.append(np.mean(loss))
    metrics.max_losses.append(np.max(difference))
    metrics.r2s.append(r2)

    return metrics


def process_data_by_gender_and_age_bounds(df, gender, age_bounds):
    try:
        df_restricted = df
        if config.separate_by_gender:
            df_restricted = df.loc[df["gender"] == gender]  # only male/female
        df_restricted = df_restricted.loc[df_restricted["boneage"] <= age_bounds[1]]
        df_restricted = df_restricted.loc[df_restricted["boneage"] >= age_bounds[0]]
        return df_restricted
    except Exception as e:
        # raise Exception
        print(e)
        return None


class Metrics(object):
    # Stores  metrics we want to keep track of during our experiments
    def __init__(self):
        # Loss = abs(prediction - true_boneage)
        self.mean_losses = []
        # This is the metric we care most about: the std of (prediction - true boneage). Note
        # that it is the same as the std of (abs(prediction - true boneage))
        self.std_losses = []
        self.max_losses = []
        self.r2s = []
        self.num_samples = []

    def reset(self):
        self.__init__()

    def update_from_local_metrics(self, local_metrics):
        self.mean_losses.append(round(np.mean(local_metrics.mean_losses), 3))
        self.std_losses.append(round(np.mean(local_metrics.std_losses), 3))
        self.max_losses.append(np.round(np.mean(local_metrics.max_losses), 3))
        self.r2s.append(np.round(np.mean(local_metrics.r2s), 3))
        self.num_samples.append(local_metrics.num_samples)

    def produce_dataframe(
        self,
    ):
        return pd.DataFrame(
            {
                "Std age difference (months)": self.std_losses,
                "Mean error": self.mean_losses,
                "Max difference": self.max_losses,
                "R^2 score": self.r2s,
                "Num samples trained on": self.num_samples,
            }
        )


def do_data_analysis_for_age_bound(df, age_bounds):

    local_metrics = Metrics()
    local_metrics.num_samples.append(0)
    # Train one model for each gender.
    # There is a break below in case we are not separating the data by genders
    # as specified by the parameter `config.separate_by_gender`
    for gender in [0, 1]:
        df_restricted = process_data_by_gender_and_age_bounds(df, gender, age_bounds)
        df_train, target_train, df_test, target_test = prepare_and_split_train_test(
            df_restricted
        )
        local_metrics.num_samples[0] += len(df_train)
        # Each call to `prepare_and_split_train_test` shuffles `df`, resulting in a different
        # train/test split. Thus we train the model many times with different splits and take the
        # average results at the end (this serves as a cross-valitation process)
        for _ in range(100):
            (
                df_train,
                target_train,
                df_test,
                target_test,
            ) = prepare_and_split_train_test(df_restricted)
            local_metrics = main_train_test_fun(
                df_train, target_train, df_test, target_test, local_metrics
            )
        # Avoid making a new iteration if we are not training separate models by gender
        if not config.separate_by_gender:
            break
    return local_metrics


def main(config):
    """This function receives a config object and performs the main routines for
    preparing the dataframe `df`, fitting a linear regression, and evaluating the results
    """
    df = pd.read_csv(config.features_df_path).iloc[
        :, 1:
    ]  # the 1: removes Unnamed: 0 -- how to avoid having it in the first place?
    df = remove_forbidden_imgs(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()

    # We will train a model (or two models if we separate by gender) for each of the age bounds in the config object.
    # E.g. id AGE_BOUNDS = [(0,10), (10,20)], we train a model for kids less than 10 years old, and another for kids
    # between 10 and 20 years old.

    # We use the Metrics object to keep track of the metrics we want to measure. We have one,
    # `global_metrics` which stores the results for each age bound. For each age bound we have
    # a `local_metrics` instantiation of the Metrics object, which measures the results obtained
    # fot the specific age bound.
    global_metrics = Metrics()
    for age_bounds in config.AGE_BOUNDS:
        local_metrics = do_data_analysis_for_age_bound(df, age_bounds)
        global_metrics.update_from_local_metrics(local_metrics)

    df_results = global_metrics.produce_dataframe()
    age_bounds_in_years = [(x[0] / 12, x[1] / 12) for x in config.AGE_BOUNDS]
    df_results["age_bounds (years)"] = age_bounds_in_years
    print(df_results)


def get_worse_performing_samples(df, losses, num_samples=20):
    # Currently NOT IN USE
    # Returns a pd.Series with the id of the `num_samples` worst performing samples
    assert df.shape[0] == len(losses), f"{df.shape[0]}, {len(losses)}"
    df["losses"] = losses
    df = df.sort_values(by="losses", axis=0)
    return df["id"].iloc[-num_samples:]


if __name__ == "__main__":
    print("Modeling with ALL FEATURES")
    main(config)

    # Below we train a model using only one relevant feature at a time.
    # This allows to find de R2 scores of each indiviual feature
    for feat in [
        x for x in config.ALL_FEATURE_NAMES if x not in ["id", "boneage", "gender"]
    ]:
        print(f"\n\nModeling with single FEATURE {feat}\n")
        config.FEATURES_FOR_DATA_ANALYSIS = ["id", "boneage", "gender", feat]
        main(config)
