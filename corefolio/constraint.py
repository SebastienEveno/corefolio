"""This module contains the Constraints classes, which are used to apply constraints to the optimization problem."""

import cvxpy as cp
from typing import List
from abc import ABC, abstractmethod


class Constraint(ABC):
    @abstractmethod
    def apply_constraint(self, variables: List[cp.Variable]) -> cp.Constraint:
        """
        Applies the constraint to the optimization problem.

        Args:
            variables (List[cp.Variable]): The decision variables.

        Returns:
            cp.Constraint: The constraint.
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

    def apply_constraint(self, variables: List[cp.Variable]) -> cp.Constraint:
        """
        Applies the constraint to the optimization problem.

        Args:
            variables (List[cp.Variable]): The decision variables.

        Returns:
            cp.Constraint: The constraint.
        """
        return cp.sum(variables) <= self._max_assets

    @property
    def max_assets(self) -> int:
        """
        Returns the maximum number of assets.

        Returns:
            int: The maximum number of assets.
        """
        return self._max_assets


class MeanConstraint(Constraint):
    def __init__(self, reference_value: float = 0.5, tolerance: float = 0.1) -> None:
        """
        Initializes the MeanConstraint with a reference value and tolerance.

        Args:
            reference_value (float): The reference value for the mean constraint.
            tolerance (float): The tolerance for the mean constraint.
        """
        self._reference_value = reference_value
        self._tolerance = tolerance

    @property
    def reference_value(self) -> float:
        """
        Returns the reference value for the mean constraint.

        Returns:
            float: The reference value for the mean constraint.
        """
        return self._reference_value

    @property
    def tolerance(self) -> float:
        """
        Returns the tolerance for the mean constraint.

        Returns:
            float: The tolerance for the mean constraint.
        """
        return self._tolerance

    def apply_constraint(self, variables: List[cp.Variable]) -> cp.Constraint:
        """
        Applies the mean constraint to the optimization problem.

        Args:
            variables (List[cp.Variable]): The decision variables.

        Returns:
            cp.Constraint: The mean constraint.
        """
        return cp.abs(cp.mean(variables) - self._reference_value) <= self._tolerance
