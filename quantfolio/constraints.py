"""This module contains the Constraints class, which is used to apply constraints to the optimization problem."""

import cvxpy as cp


class Constraints:
    """A class used to apply constraints to the optimization problem."""
    @staticmethod
    def apply_constraints(variables: list[int], max_assets: int = 5):
        """Apply constraints to the optimization problem."""
        constraints = []
        constraints.append(cp.sum(variables) <= max_assets)
        return constraints
