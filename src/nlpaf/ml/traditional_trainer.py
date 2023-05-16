from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold,
)
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import classification_report


def dummy_train_test(
    X_train, y_train, X_test, y_test, strategy="most_frequent"
):
    clf = DummyClassifier(strategy=strategy, random_state=1)
    """
    “stratified”: generates predictions by respecting the training set’s class distribution.

    “most_frequent”: always predicts the most frequent label in the training set.

    “prior”: always predicts the class that maximizes the class prior (like “most_frequent”) and predict_proba returns the class prior.

    “uniform”: generates predictions uniformly at random.

    “constant”: always predicts a constant label that is provided by the user. This is useful for metrics that evaluate a non-majority class
    """

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1_dic = {}

    f1_dic["macro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="macro"), 2
    )
    f1_dic["micro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="micro"), 2
    )

    classes = list(np.unique(y_train))
    for label in classes:
        f1_dic[label] = round(
            f1_score(
                y_pred=y_pred, y_true=y_test, average=None, labels=[label]
            )[0],
            2,
        )

    return f1_dic


def train_save(
    X_train, y_train, pkl_filename, algorithm="svm", details=False, params={}
):
    print("training and saving")
    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    print("running ", algorithm, " with params ", str(params))
    if algorithm == "svm":
        cw = params["class_weight"] if "class_weight" in params else "balanced"
        c = params["C"] if "C" in params else 1
        clf = SVC(kernel="linear", class_weight=cw, C=c)
    elif algorithm == "randomforest":
        estimator = params["n_estimators"] if "n_estimators" in params else 1
        max_depth = params["max_depth"] if "max_depth" in params else 2
        clf = RandomForestClassifier(
            n_estimators=estimator, max_depth=max_depth, random_state=1
        )
    else:
        print(
            "algorithm not supported! valid values are: svm and randomforest"
        )
        return

    clf.fit(X_train, y_train)
    with open(pkl_filename, "wb") as f:
        pickle.dump(clf, f)


def train_test(
    X_train, y_train, X_test, y_test, algorithm="svm", details=False, params={}
):
    print("training and testing")
    # 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
    if details:
        print("running ", algorithm, " with params ", str(params))
    if algorithm == "svm":
        cw = params["class_weight"] if "class_weight" in params else "balanced"
        c = params["C"] if "C" in params else 1
        clf = SVC(kernel="linear", class_weight=cw, C=c)
    elif algorithm == "randomforest":
        estimator = params["n_estimators"] if "n_estimators" in params else 1
        max_depth = params["max_depth"] if "max_depth" in params else 2
        clf = RandomForestClassifier(
            n_estimators=estimator, max_depth=max_depth, random_state=1
        )
    else:
        print(
            "algorithm not supported! valid values are: svm and randomforest"
        )
        return

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    if details:
        print("Confusion Matrix:")

        print(confusion_matrix(y_test, y_pred))

        print()
        print("Accuracy: ", round(accuracy_score(y_test, y_pred), 2))
        print("Report:")
        print(classification_report(y_test, y_pred))
    f1_dic = {}

    f1_dic["macro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="macro"), 2
    )
    f1_dic["micro"] = round(
        f1_score(y_pred=y_pred, y_true=y_test, average="micro"), 2
    )

    f1_dic["accuracy"] = round(accuracy_score(y_test, y_pred), 2)
    # f1_dic['no-effect']= round(f1_score(y_pred=y_pred, y_true=y_test, average=None, labels=['no-effect'])[0], 2)
    classes = list(np.unique(y_train))
    for label in classes:
        f1_dic[label] = round(
            f1_score(
                y_pred=y_pred, y_true=y_test, average=None, labels=[label]
            )[0],
            2,
        )

    return f1_dic


def svc_param_gridsearch(X, y, nfolds_or_division):
    print("performing gridsearch")
    Cs = [0.01, 0.1, 1, 10, 100]
    param_grid = {"C": Cs, "class_weight": ["balanced"]}  # , 'gamma' : gammas}
    grid_search = GridSearchCV(
        SVC(kernel="linear"),
        param_grid,
        cv=nfolds_or_division,
        n_jobs=-1,
        scoring="f1_macro",
    )

    grid_search.fit(X, y)

    return grid_search.best_params_  # , grid_search.best_score_


def randomforest_param_gridsearch(X, y, nfolds_or_division):
    estimators = range(1, 41)
    param_grid = {"n_estimators": estimators}  # , 'gamma' : gammas}
    grid_search = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, class_weight="balanced"),
        param_grid,
        n_jobs=-1,
        scoring="f1_macro",
    )

    grid_search.fit(X, y)

    return grid_search.best_params_  # , grid_search.best_score_


def train_ensemble(X_train, X_test, y_train, y_test):
    # calculate class weights
    weights = class_weight.compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(zip(np.unique(y_train), weights))
    # define the individual models
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight=class_weights
    )
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm = SVC(probability=True, random_state=42, class_weight=class_weights)

    # define the ensemble model
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("svm", svm)], voting="soft"
    )

    # define the balanced bagging classifier for resampling
    bagging = BalancedBaggingClassifier(
        estimator=ensemble,
        sampling_strategy="auto",
        replacement=False,
        random_state=42,
    )

    # define the cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # fit and evaluate the resampled ensemble model with cross-validation
    ros = RandomOverSampler(sampling_strategy="minority", random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    scores = cross_val_score(
        bagging, X_resampled, y_resampled, scoring="f1_macro", cv=cv
    )
    print("Cross-validation scores:", scores)
    print("Mean F1 score:", np.mean(scores))

    result = {}
    result["Cross-validation scores:"] = scores
    result["Mean F1 score:"] = np.mean(scores)

    # fit the resampled ensemble model on the entire training set
    bagging.fit(X_resampled, y_resampled)

    # make predictions on the test set
    y_pred = bagging.predict(X_test)

    # evaluate the performance of the ensemble model on the test set
    result["report"] = classification_report(y_test, y_pred)
    print(result["report"])
