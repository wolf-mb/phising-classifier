import pandas as pd
import numpy as np

class DataTransformer:
    """
    Handles data cleaning, feature engineering, and validation.
    This module makes your pipeline robust and reusable.
    """
    def __init__(self):
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates and handles missing values."""
        # Drop duplicates to prevent data leakage
        df = df.drop_duplicates()
        # Fill missing values with 0 (Neutral)
        df = df.fillna(0)
        return df

    def transform_to_features(self, df: pd.DataFrame) -> tuple:
        """Separates features and targets cleanly."""
        target_column = 'Result' if 'Result' in df.columns else df.columns[-1]
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y