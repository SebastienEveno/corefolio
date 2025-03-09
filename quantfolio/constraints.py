import cvxpy as cp


class Constraints:
    @staticmethod
    def apply_constraints(variables: list[int], max_assets: int = 5):
        constraints = []
        constraints.append(cp.sum(variables) <= max_assets)
        return constraints
