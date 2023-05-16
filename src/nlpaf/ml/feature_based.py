import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from functools import reduce  # python 3
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

from pathlib import Path
import dataclasses
from nlpaf.ml import preprocess, traditional_trainer


@dataclasses.dataclass
class FeatureBasedML:
    y_col: str
    dataset: pd.DataFrame
    split_label_name: str = "split_label"
    training_cols: list = None
    remove_outliers: bool = True
    normalize: bool = False
    normalizing_method: str = preprocess._NORMALIZE_METHOD_STANDARD_

    def __post_init__(self):
        self.normalized = False
        self.X_train, self.y_train, self.X_test, self.y_test = self.set_x_y()
        self.run_normalize()

    def sample(self, type_: str):
        # SAMPLING
        if type_ == "OVER":
            self.X_train, self.y_train = preprocess.over_sampler(
                self.X_train,
                self.y_train,
                sampling_strategy_=self.sampling_strategy,
            )
        elif type_ == "UNDER":
            self.X_train, self.y_train = preprocess.under_sampler(
                self.X_train,
                self.y_train,
                sampling_strategy_=self.sampling_strategy,
            )
        elif type_ == "SMOTE":
            self.X_train, self.y_train = preprocess.smote(
                self.X_train,
                self.y_train,
                sampling_strategy_=self.sampling_strategy,
            )

    def run_normalize(self):
        if self.normalize and not self.normalized:
            self.X_train, self.X_test = preprocess.normalize(
                self.X_train,
                self.X_test,
                normalizing_method=self.normalizing_method,
            )
            self.normalized = True

    def set_x_y(self):
        df = self.dataset.fillna(0).copy()
        df_train = df[df[self.split_label_name] == "train"]
        df_test = df[df[self.split_label_name] == "test"]

        # Remove outliers:
        # cols= get_features_map(df_train).values()
        # cols= reduce(lambda x, y: x + y, cols)
        if self.remove_outliers:
            print("removing outliers by clipping values...")

            # df_train_features= df_train[cols]
            df_train, df_test = preprocess.clip_outliers(
                df_train, df_test, lower_percentile=1, upper_percentile=99
            )

        # X Y
        print("getting X y data...")
        training_cols = (
            self.training_cols
            if self.training_cols is not None
            else df_train.columns.tolist()
        )
        if self.y_col in training_cols:
            training_cols.remove(self.y_col)
        X_train = df_train[training_cols]
        y_train = df_train[self.y_col].values

        X_test = df_test[training_cols]
        y_test = df_test[self.y_col].values

        print("end of get_x_y.")
        return X_train, y_train, X_test, y_test

    def train_xfold(self, folds=5):
        best_params = traditional_trainer.svc_param_gridsearch(
            self.X_train, self.y_train, nfolds_or_division=folds
        )

        result = traditional_trainer.train_test(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            params=best_params,
        )

        result["params"] = best_params
        return result

    def train_baseline(self, strategy: str = "uniform"):
        return traditional_trainer.dummy_train_test(
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            strategy=strategy,
        )
