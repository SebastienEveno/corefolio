"""Tests for the Universe class."""

import pytest
import pandas as pd

from quantfolio.universe import Universe


def test_universe_initialization():
    """Test the initialization of the Universe class."""
    data = pd.DataFrame({"ID": [1, 2, 3], "value": [10, 20, 30]})
    universe = Universe(data)
    assert universe.number_of_assets == 3


def test_universe_nan_values():
    """Test that an exception is raised if the DataFrame contains NaN values."""
    data = pd.DataFrame({"ID": [1, 2, None], "value": [10, 20, 30]})
    with pytest.raises(Exception, match="DataFrame contains NaN values."):
        Universe(data)


def test_universe_duplicate_ids():
    """Test that an exception is raised if the DataFrame contains duplicate IDs."""
    data = pd.DataFrame({"ID": [1, 2, 2], "value": [10, 20, 30]})
    with pytest.raises(Exception, match="DataFrame contains duplicate IDs."):
        Universe(data)


def test_universe_from_dataframe():
    """Test the from_dataframe class method."""
    data = pd.DataFrame({"ID": [1, 2, 3], "value": [10, 20, 30]})
    universe = Universe.from_dataframe(data)
    assert universe.to_dataframe().equals(data)


def test_universe_custom_id_column():
    """Test the initialization of the Universe class with a custom ID column."""
    data = pd.DataFrame({"Asset_ID": [1, 2, 3], "value": [10, 20, 30]})
    universe = Universe.from_dataframe(data, id_column="Asset_ID")
    assert universe.number_of_assets == 3
