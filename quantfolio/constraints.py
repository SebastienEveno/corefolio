import cvxpy as cp


class Constraints:
    @staticmethod
    def apply_constraints(variables: list[int], max_assets: int = 5, solver: str = "cvxpy"):
        constraints = []
        if solver == "cvxpy":
            constraints.append(cp.sum(variables) <= max_assets)
        elif solver == "pyomo":
            constraints.append(sum(variables) <= max_assets)
        return constraints
