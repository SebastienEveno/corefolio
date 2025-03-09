import cvxpy as cp

from quantfolio.constraints import Constraints


def test_apply_constraints():
    x = cp.Variable(3, boolean=True)
    constraints = Constraints.apply_constraints(
        x, max_assets=2)
    assert len(constraints) == 1
