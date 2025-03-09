import cvxpy as cp
import pyomo.environ as pyo

from quantfolio.constraints import Constraints


def test_apply_constraints_cvxpy():
    x = cp.Variable(3, boolean=True)
    constraints = Constraints.apply_constraints(
        x, max_assets=2, solver="cvxpy")
    assert len(constraints) == 1


def test_apply_constraints_pyomo():
    x = [pyo.Var(domain=pyo.Binary) for _ in range(3)]
    constraints = Constraints.apply_constraints(
        x, max_assets=2, solver="pyomo")
    assert len(constraints) == 1
