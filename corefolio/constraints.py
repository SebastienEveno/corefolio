"""This module contains the Constraints class, which is used to apply constraints to the optimization problem."""

import cvxpy as cp
from typing import List


class Constraints:
    def __init__(self, max_assets: int = 5) -> None:
        """
        Initializes the Constraints with a maximum number of assets.

        Args:
            max_assets (int): The maximum number of assets to select.
        """
        self._max_assets = max_assets

    def apply_constraints(self, variables: List[cp.Variable]) -> List[cp.Constraint]:
        """
        Applies the constraints to the optimization problem.

        Args:
            variables (List[cp.Variable]): The decision variables.

        Returns:
            List[cp.Constraint]: The list of constraints.
        """
        constraints = []
        constraints.append(cp.sum(variables) <= self._max_assets)
        return constraints

    @property
    def max_assets(self) -> int:
        """
        Returns the maximum number of assets.

        Returns:
            int: The maximum number of assets.
        """
        return self._max_assets
