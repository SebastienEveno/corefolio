"""This module contains the Constraints classes, which are used to apply constraints to the optimization problem."""

import cvxpy as cp
from typing import List, Optional
from abc import ABC, abstractmethod
import pandas as pd


class Constraint(ABC):
    @abstractmethod
    def apply_constraint(self, variables: List[cp.Variable], df: pd.DataFrame) -> List[cp.Constraint]:
        """
        Applies the constraint to the optimization problem.

        Args:
            variables (List[cp.Variable]): The decision variables.
            df (pd.DataFrame): The DataFrame containing asset data.

        Returns:
            List[cp.Constraint]: The list of constraints.
        """
        pass


class MaxAssetsConstraint(Constraint):
    def __init__(self, max_assets: int = 5) -> None:
        """
        Initializes the MaxAssetsConstraint with a maximum number of assets.

        Args:
            max_assets (int): The maximum number of assets to select.
        """
        self._max_assets = max_assets

    def apply_constraint(self, variables: List[cp.Variable], df: pd.DataFrame) -> List[cp.Constraint]:
        """
        Applies the constraint to the optimization problem.

        Args:
            variables (List[cp.Variable]): The decision variables.
            df (pd.DataFrame): The DataFrame containing asset data.

        Returns:
            List[cp.Constraint]: The list of constraints.
        """
        return [cp.sum(variables) <= self._max_assets]

    @property
    def max_assets(self) -> int:
        """
        Returns the maximum number of assets.

        Returns:
            int: The maximum number of assets.
        """
        return self._max_assets


class MeanConstraint(Constraint):
    def __init__(self, column_name: str, tolerance: float = 0.01, min_value: Optional[float] = None, max_value: Optional[float] = None) -> None:
        """
        Initializes the MeanConstraint with a column name, tolerance, and optional minimum and maximum values.

        Args:
            column_name (str): The column name to be used for the mean constraint.
            tolerance (float): The tolerance for the mean constraint.
            min_value (Optional[float]): The minimum value for the mean constraint.
            max_value (Optional[float]): The maximum value for the mean constraint.
        """
        self._column_name = column_name
        self._tolerance = tolerance
        self._min_value = min_value
        self._max_value = max_value

    def apply_constraint(self, variables: List[cp.Variable], df: pd.DataFrame) -> List[cp.Constraint]:
        """
        Applies the mean constraint to the optimization problem.

        Args:
            variables (List[cp.Variable]): The decision variables.
            df (pd.DataFrame): The DataFrame containing asset data.

        Returns:
            List[cp.Constraint]: The list of constraints.
        """
        constraints = []
        if pd.api.types.is_numeric_dtype(df[self._column_name]):
            mean_value = df[self._column_name].mean()
            column_values = df[self._column_name].values
            selected_sum = cp.sum(cp.multiply(variables, column_values))
            selected_count = cp.sum(variables)

            min_value = self._min_value if self._min_value is not None else mean_value - self._tolerance
            max_value = self._max_value if self._max_value is not None else mean_value + self._tolerance

            constraints.append(selected_sum >= selected_count * min_value)
            constraints.append(selected_sum <= selected_count * max_value)
        else:
            categories = df[self._column_name].unique()
            for category in categories:
                category_mask = (df[self._column_name] ==
                                 category).astype(float)
                category_frequency = category_mask.mean()
                selected_sum = cp.sum(cp.multiply(
                    variables, category_mask))
                selected_count = cp.sum(variables)

                min_value = self._min_value if self._min_value is not None else category_frequency - self._tolerance
                max_value = self._max_value if self._max_value is not None else category_frequency + self._tolerance

                constraints.append(selected_sum >= selected_count * min_value)
                constraints.append(selected_sum <= selected_count * max_value)

        return constraints

    @property
    def column_name(self) -> str:
        """
        Returns the column name for the mean constraint.

        Returns:
            str: The column name for the mean constraint.
        """
        return self._column_name

    @property
    def tolerance(self) -> float:
        """
        Returns the tolerance for the mean constraint.

        Returns:
            float: The tolerance for the mean constraint.
        """
        return self._tolerance

    @property
    def min_value(self) -> Optional[float]:
        """
        Returns the minimum value for the mean constraint.

        Returns:
            Optional[float]: The minimum value for the mean constraint.
        """
        return self._min_value

    @property
    def max_value(self) -> Optional[float]:
        """
        Returns the maximum value for the mean constraint.

        Returns:
            Optional[float]: The maximum value for the mean constraint.
        """
        return self._max_value
