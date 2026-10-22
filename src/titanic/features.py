"""Feature engineering for the Titanic competition.

Two kinds of steps live here:
- `add_features` is stateless: each row is transformed on its own, nothing is learned from the data.
- `TitleAgeImputer` and `GroupSurvivalEncoder` learn from the data in `fit`, so they must sit inside a
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


def group_keys(X):
    """Ticket and family (surname + fare) keys used to find passengers travelling together."""
    surname = X["Name"].str.split(",").str[0].str.strip()
    family = surname + "_" + X["Fare"].round(2).astype(str)
    return X["Ticket"].astype(str).to_numpy(), family.to_numpy()


class GroupSurvivalEncoder(BaseEstimator, TransformerMixin):
    """Survival rate of the *other* known passengers sharing a ticket or a family (surname + fare).

    Adds `GroupSurvival` (rate, `default` when no other member's label is known) and `GroupKnown` (0/1).
    The feature is built from training labels, so it leaks unless two rules hold:
    - it is fitted inside the Pipeline, on the training folds only;
    - on the training rows each passenger's own label is left out (leave-one-out), otherwise the feature
      contains the answer. Like scikit-learn's TargetEncoder, `fit_transform` therefore differs from
      `fit(...).transform(...)`: only `transform` (used for new passengers) sees all stored labels.
    """

    def __init__(self, default=0.5):
        self.default = default

    def fit(self, X, y):
        self.tickets_, self.families_ = group_keys(X)
        self.labels_ = np.asarray(y, dtype=float)
        return self

    def transform(self, X):
        return self._encode(X, leave_one_out=False)

    def fit_transform(self, X, y):
        return self.fit(X, y)._encode(X, leave_one_out=True)

    def _encode(self, X, leave_one_out):
        tickets, families = group_keys(X)
        rates = np.full(len(X), self.default)
        known = np.zeros(len(X), dtype=int)
        for i, (ticket, family) in enumerate(zip(tickets, families)):
            members = (self.tickets_ == ticket) | (self.families_ == family)
            if leave_one_out:
                members[i] = False  # row i of X is training row i: drop the passenger's own label
            if members.any():
                rates[i] = self.labels_[members].mean()
                known[i] = 1
        out = X.copy()
        out["GroupSurvival"] = rates
        out["GroupKnown"] = known
        return out


def make_preprocessor(group_survival=False):
    """Full preprocessing: engineered features -> age imputation -> scaling and one-hot encoding.

    With `group_survival=True` the `GroupSurvivalEncoder` step is added and its two columns are used.
    """
    numeric = NUMERIC_FEATURES + (["GroupSurvival"] if group_survival else [])
    binary = BINARY_FEATURES + (["GroupKnown"] if group_survival else [])
    columns = ColumnTransformer(
        [
            ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("binary", "passthrough", binary),
            ("categorical",
             make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
             CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )
    steps = [
        ("engineer", FunctionTransformer(add_features)),
        ("impute_age", TitleAgeImputer()),
    ]
    if group_survival:
        steps.append(("group_survival", GroupSurvivalEncoder()))
    steps.append(("columns", columns))
    return Pipeline(steps)
