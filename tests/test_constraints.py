"""Tests for the constraints module."""

import cvxpy as cp
import pandas as pd

from corefolio.constraint import MaxAssetsConstraint, MeanConstraint


def test_apply_max_assets_constraint():
    """
    Test the apply_constraint method of the MaxAssetsConstraint class.
    Ensures that the correct constraint is applied based on max_assets.
    """
    x = cp.Variable(3, boolean=True)
    constraint = MaxAssetsConstraint(max_assets=2)
    applied_constraints = constraint.apply_constraint(x, pd.DataFrame())[0]
    assert applied_constraints.args[1].value == 2


def test_max_assets_property():
    """
    Test the max_assets property of the MaxAssetsConstraint class.
    Ensures that the max_assets property returns the correct value.
    """
    constraint = MaxAssetsConstraint(max_assets=5)
    assert constraint.max_assets == 5


def test_apply_mean_constraint_numerical():
    """
    Test the apply_constraint method of the MeanConstraint class with numerical data.
    Ensures that the correct constraints are applied based on the mean and tolerance.
    """
    df = pd.DataFrame({"value": [10, 20, 30, 40]})
    x = cp.Variable(len(df), boolean=True)
    constraint = MeanConstraint(column_name="value", tolerance=0.01)
    applied_constraints = constraint.apply_constraint(x, df)
    assert len(applied_constraints) == 2


def test_apply_mean_constraint_categorical():
    """
    Test the apply_constraint method of the MeanConstraint class with categorical data.
    Ensures that the correct constraints are applied based on the category frequencies and tolerance.
    """
    df = pd.DataFrame({"category": ["A", "A", "B", "B"]})
    x = cp.Variable(len(df), boolean=True)
    constraint = MeanConstraint(column_name="category", tolerance=0.01)
    applied_constraints = constraint.apply_constraint(x, df)
    assert len(applied_constraints) == 4


def test_apply_mean_constraint_min_max():
    """
    Test the apply_constraint method of the MeanConstraint class with numerical data and absolute min/max values.
    Ensures that the correct constraints are applied based on the specified min and max values.
    """
    df = pd.DataFrame({"value": [10, 20, 30, 40]})
    x = cp.Variable(len(df), boolean=True)
    constraint = MeanConstraint(
        column_name="value", min_value=15, max_value=35)
    applied_constraints = constraint.apply_constraint(x, df)
    assert len(applied_constraints) == 2


def test_mean_constraint_properties():
    """
    Test the properties of the MeanConstraint class.
    Ensures that the properties return the correct values.
    """
    constraint = MeanConstraint(
        column_name="value", tolerance=0.01, min_value=15, max_value=35)
    assert constraint.column_name == "value"
    assert constraint.tolerance == 0.01
    assert constraint.min_value == 15
    assert constraint.max_value == 35
