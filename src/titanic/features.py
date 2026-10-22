"""Feature engineering for the Titanic competition.

Two kinds of steps live here:
- `add_features` is stateless: each row is transformed on its own, nothing is learned from the data.
- `TitleAgeImputer` learns statistics (median age per title) in `fit`, so it must sit inside a
  scikit-learn Pipeline to be fitted on the training folds only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

COMMON_TITLES = ["Mr", "Mrs", "Miss", "Master"]
TITLE_ALIASES = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}

NUMERIC_FEATURES = ["Age", "LogFare"]
BINARY_FEATURES = ["AgeMissing", "HasCabin"]
CATEGORICAL_FEATURES = ["Pclass", "Gender", "Title", "FamilyType", "Embarked"]


def extract_title(names):
    """'Braund, Mr. Owen Harris' -> 'Mr'. Uncommon titles are grouped as 'Rare'."""
    titles = names.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip().replace(TITLE_ALIASES)
    return titles.where(titles.isin(COMMON_TITLES), "Rare")


def add_features(df):
    """Add engineered columns derived row by row from the raw Kaggle columns."""
    out = df.copy()
    out["Title"] = extract_title(out["Name"])
    out["AgeMissing"] = out["Age"].isna().astype(int)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    out["LogFare"] = np.log1p(out["Fare"])
    family_size = out["SibSp"] + out["Parch"] + 1
    out["FamilyType"] = pd.cut(family_size, bins=[0, 1, 4, 11],
                               labels=["alone", "small", "large"]).astype(str)
    return out


class TitleAgeImputer(BaseEstimator, TransformerMixin):
    """Fill missing Age with the median age of the passenger's title, learned in `fit`."""

    def fit(self, X, y=None):
        self.medians_ = X.groupby("Title")["Age"].median()
        self.global_median_ = X["Age"].median()
        return self

    def transform(self, X):
        out = X.copy()
        out["Age"] = out["Age"].fillna(out["Title"].map(self.medians_)).fillna(self.global_median_)
        return out


def make_preprocessor():
    """Full preprocessing: engineered features -> age imputation -> scaling and one-hot encoding."""
    columns = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
            ("categorical",
             make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
             CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("engineer", FunctionTransformer(add_features)),
        ("impute_age", TitleAgeImputer()),
        ("columns", columns),
    ])
