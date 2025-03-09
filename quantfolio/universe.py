"""This module contains the Universe class."""

import pandas as pd


class Universe:
    """A class used to represent the universe of assets."""

    def __init__(self, df: pd.DataFrame, id_column: str = "ID") -> None:
        """Initialize the Universe class."""
        if df.isna().any().any():
            raise Exception("DataFrame contains NaN values.")
        if df[id_column].duplicated().any():
            raise Exception("DataFrame contains duplicate IDs.")

        self.df = df
        self.id_column = id_column
        self.number_of_assets = df[id_column].nunique()

    def get_data(self) -> pd.DataFrame:
        """Returns the Universe data as a DataFrame."""
        return self.df

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, id_column: str = "ID"):
        """Creates a Universe instance from a DataFrame."""
        return cls(df, id_column)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the Universe data as a DataFrame."""
        return self.df
