"""This module contains the Universe class."""

import pandas as pd


class Universe:
    def __init__(self, df: pd.DataFrame, id_column: str = "ID") -> None:
        if df.isna().any().any():
            raise Exception("DataFrame contains NaN values.")

        if id_column not in df.columns:
            df[id_column] = range(1, len(df) + 1)

        if df[id_column].duplicated().any():
            raise Exception("DataFrame contains duplicate IDs.")

        self._df = df
        self._id_column = id_column
        self._number_of_assets = df[id_column].nunique()

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, id_column: str = "ID"):
        """Creates a Universe instance from a DataFrame."""
        return cls(df, id_column)

    def to_dataframe(self) -> pd.DataFrame:
        """Returns the Universe data as a DataFrame while keeping it protected from modification."""
        return self._df.copy()

    @property
    def df(self) -> pd.DataFrame:
        """Returns the Universe data as a DataFrame while keeping it protected from modification."""
        return self.to_dataframe()

    @property
    def id_column(self) -> str:
        """Returns the ID column name."""
        return self._id_column

    @property
    def number_of_assets(self) -> int:
        """Returns the number of assets in the Universe."""
        return self._number_of_assets
