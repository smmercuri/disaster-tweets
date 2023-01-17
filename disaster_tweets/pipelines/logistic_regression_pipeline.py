from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def logistic_regression_pipeline(
    cat_cols: List[str] = None, verbose: bool = True
) -> Pipeline:
    """
    Simple logistic regression sklearn pipeline designed to predict base only on categorical features, such as
    `keyword` and/or `location`. Consists of a one-hot encoding preprocessing step and a logistic regression modelling step.

    Parameters
    ----------
    cat_cols : List[str], optional
        Which categorical columns to use, by default ['keyword']
    verbose : bool, optional
        Whether to print training information, by default True

    Returns
    -------
    Pipeline
        sklearn pipeline consisting of a one-hot encoding preprocessing step and logistic regression modelling step.
    """
    cat_cols = ["keyword"] if cat_cols is None else cat_cols
    preprocessing = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                cat_cols,
            )
        ],
        remainder="drop",
    )
    steps = [("preprocessing", preprocessing), ("clf", LogisticRegression())]
    return Pipeline(steps=steps, verbose=verbose)
