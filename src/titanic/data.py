"""Load the Kaggle Titanic CSV files with the project's column names."""
import pandas as pd

# The raw Kaggle files use different names for some columns; the project only uses the names on the right.
RAW_TO_PROJECT_COLUMNS = {"Sex": "Gender"}


def load_raw(path):
    """Read a Kaggle Titanic CSV file and rename columns to the project's names."""
    return pd.read_csv(path).rename(columns=RAW_TO_PROJECT_COLUMNS)
