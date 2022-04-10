This is an ongoing project where our goal is to extract interpretable features from hand radiographies in order to create a simple predictive model for bone age

The data can be found in <https://www.kaggle.com/kmader/rsna-bone-age>

## Usage

### Fiting linear regressions
Running `python3 scripts/data_analysis.py`will fit a linear regression using the data from `data/features_df.csv`, restricting to the features specified in `config.FEATURES_FOR_DATA_ANALYSIS` (the features `id`, `boneage` and `gender` are not used though) (see `config.py`). It also fits a linear regression for each feature, so we can see the R2 scores of each individual features. The output looks like:

    Modeling with ALL FEATURES
       Std age difference (months)  Mean error  Max difference  R^2 score Num samples trained on age_bounds (years)
    0                       10.364      12.244          26.581      0.695                   [22]         (0.0, 6.0)
    1                        8.416      11.217          34.604      0.511                  [213]        (6.0, 12.0)
    2                        7.306       8.969          20.158      0.248                  [243]       (12.0, 20.0)
    3                       10.953      14.258          51.260      0.730                  [452]        (0.0, 20.0)
    
    Modeling with single FEATURE max_purple_diameter
    
       Std age difference (months)  Mean error  Max difference  R^2 score Num samples trained on age_bounds (years)
    0                        6.635      10.759          18.671      0.452                   [26]         (0.0, 6.0)
    1                        9.827      13.693          40.005      0.298                  [216]        (6.0, 12.0)
    2                        7.619       9.274          21.687      0.176                  [244]       (12.0, 20.0)
    3                       14.088      19.312          62.704      0.495                  [453]        (0.0, 20.0)
    
    [...]

### Obtaining `features_df.csv`

The dataset `features_df.csv` is obtained by running <python3 scripts/get_features.py>. This command will create all the features listed in `config.ALL_FEATURES`. The code expects that, for each feature, in `config.ALL_FEATURES`, the class `Hand` has a getter method called `get_<feature_name>`.

New features can be added/modified by adding/modifying the corresponding method in the class Hand

The features are generated using the following information:
1. The contours of the bones, which we outlined privately.
2. Google's mediapipe library, which allows to detect hand landmarks: <https://google.github.io/mediapipe/solutions/hands>

### Creating scatterplots for the features

Run `python3 scripts/scatterplots_of_features.py`. This will take the feature DataFrame created with `get_features.py` and draw scatterplots for each feature. The graphs are saved in `bone-age/data/feature-scatterplots`
