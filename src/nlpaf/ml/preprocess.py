from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import numpy as np
import pandas as pd


# CONSTANTS #

_NORMALIZE_METHOD_STANDARD_ = "standard"
_NORMALIZE_METHOD_LOG_ = "log"
_NORMALIZE_METHOD_SQRT_ = "sqrt"
_NORMALIZE_METHOD_MINMAX_ = "minmax"


# Outliers
def clip_outliers(
    df_train, df_test=None, lower_percentile=1, upper_percentile=99
):
    def _get_upper_lower_df(train_df, lower_percentile=1, upper_percentile=99):
        boundaries = np.percentile(
            train_df, [lower_percentile, upper_percentile], axis=0
        )
        boundaries_df = pd.DataFrame(boundaries, columns=train_df.columns)
        return boundaries_df

    def _apply_clip_col(col, upper_lower_df):
        if col.name in upper_lower_df.columns:
            col = np.clip(
                col,
                max(upper_lower_df[col.name].values),
                min(upper_lower_df[col.name].values),
            )

        return col

    print("getting only numeric features from the training set...")
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numeric_df = df_train.select_dtypes(include=numerics)
    print(
        "There are {}  numeric features out of {}".format(
            str(len(numeric_df.columns)), str(len(df_train.columns))
        )
    )
    boundaries_df = _get_upper_lower_df(
        numeric_df,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    df_train = df_train.apply(_apply_clip_col, args=(boundaries_df,), axis=0)
    if df_test is not None:
        df_test = df_test.apply(_apply_clip_col, args=(boundaries_df,), axis=0)

    return df_train, df_test


def normalize(X_train, X_test, normalizing_method=_NORMALIZE_METHOD_STANDARD_):
    print("Normalizing by using {} scaler...".format(normalizing_method))

    if normalizing_method == _NORMALIZE_METHOD_STANDARD_:
        print("Normalizing by using standard scaler...")
        scaler = StandardScaler(copy=True, with_mean=False)
        scaler.fit(X_train)
        X_train = pd.DataFrame(
            scaler.transform(X_train), columns=X_train.columns
        )
        X_test = (
            pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            if (X_test is not None)
            else None
        )
    elif normalizing_method == _NORMALIZE_METHOD_LOG_:
        X_train = np.log(X_train + 1)
        X_test = np.log(X_test + 1) if (X_test is not None) else None
    elif normalizing_method == _NORMALIZE_METHOD_SQRT_:
        X_train = np.sqrt(X_train + (2 / 3))
        X_test = np.sqrt(X_test + (2 / 3)) if (X_test is not None) else None
    elif normalizing_method == _NORMALIZE_METHOD_MINMAX_:
        scaler = MinMaxScaler(copy=True)
        scaler.fit(X_train)

        X_train = pd.DataFrame(
            scaler.transform(X_train), columns=X_train.columns
        )
        X_test = (
            pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
            if (X_test is not None)
            else None
        )
    else:
        print(
            "data is not normalized. method is not supported. Choose one of: standard, minmax, log, sqrt"
        )
    print("normalized")
    return X_train, X_test


def under_sampler(x, y, sampling_strategy_="not minority"):
    """Sampling strategy could be specifically the number of examples in the minority class divided by the number of examples in the majority class"""
    rus = RandomUnderSampler(
        random_state=42, sampling_strategy_=sampling_strategy_
    )
    x_under, y_under = rus.fit_resample(x, y)
    print(Counter(y))
    return x_under, y_under


# Applies Random Undersampling
def over_sampler(x, y, sampling_strategy_="not majority"): 
    """the minority class was oversampled to have half the number of examples as the majority class
    """
    ros = RandomOverSampler(
        random_state=42, sampling_strategy_=sampling_strategy_
    )
    x_, y_ = ros.fit_resample(x, y)
    print(Counter(y))
    return x_, y_ 


# Applies Synthetic Data Augmentation through SMOTE
def smote(x, y, sampling_strategy_: str = "not majority"):

    smote = SMOTE(random_state=42, sampling_strategy_=sampling_strategy_)
    x_, y_ = smote.fit_resample(x, y)
    print(Counter(y))
    return x_, y_
