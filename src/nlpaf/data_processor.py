import pandas as pd
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


def undersample(
    df,
    replacement: bool = False,
    random_state: int = 42,
    sampling_strategy: str = "not minority",
):
    col = df.columns
    features = col.tolist()
    feature = features[:-1]
    target = features[-1]
    X = df.loc[:, feature]
    y = df.loc[:, target]

    undersample = RandomUnderSampler(
        replacement=replacement,
        random_state=random_state,
        sampling_strategy=sampling_strategy,
    )
    X_under, y_under = undersample.fit_resample(X, y)

    print(f"Resampled dataset shape {Counter(y_under)}")
    return pd.concat([X_under, y_under], axis=1)
