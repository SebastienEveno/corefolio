"""Tests for the constraints module."""

import cvxpy as cp

from corefolio.constraint import Constraint


def test_apply_constraint():
    x = cp.Variable(3, boolean=True)
    constraint = Constraint(max_assets=2)
    applied_constraints = constraint.apply_constraint(x)
    assert applied_constraints.args[1].value == 2


def test_max_assets_property():
    constraint = Constraint(max_assets=5)
    assert constraint.max_assets == 5
