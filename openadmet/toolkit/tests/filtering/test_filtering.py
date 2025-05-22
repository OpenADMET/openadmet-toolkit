import pandas as pd
import pytest

from openadmet.toolkit.filtering.filter_base import min_max_filter
from openadmet.toolkit.filtering.filter_base import mark_or_remove
from openadmet.toolkit.filtering.filter_base import BaseFilter

from openadmet.toolkit.filtering.physchem_filters import pKaFilter
from openadmet.toolkit.filtering.physchem_filters import logPFilter
from openadmet.toolkit.filtering.physchem_filters import SMARTSFilter
from openadmet.toolkit.filtering.physchem_filters import SMARTSProximityFilter

from openadmet.toolkit.tests.datafiles import filtering_file

@pytest.fixture()
def clogp_data():
    df = pd.DataFrame({
        "clogp": [-2.0, 0.5, 1.5, 7.0, 8.0],
        "test_mark": [False] * 5
    })
    return df

@pytest.fixture()
def test_data():
    return filtering_file

def test_min_max_filter(clogp_data):
    min_threshold = -1.0
    max_threshold = 6.0

    # Test with both min_threshold and max_threshold
    min_max_filter(
        df=clogp_data,
        property="clogp",
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        mark_column="test_mark"
    )
    assert list(clogp_data["test_mark"]) == [False, True, True, False, False]

    # Test with only max_threshold
    min_max_filter(
        df=clogp_data,
        property="clogp",
        min_threshold=None,
        max_threshold=max_threshold,
        mark_column="test_mark"
    )
    assert list(clogp_data["test_mark"]) == [True, True, True, False, False]

    # Test with only min_threshold
    min_max_filter(
        df=clogp_data,
        property="clogp",
        min_threshold=min_threshold,
        max_threshold=None,
        mark_column="test_mark"
    )
    assert list(clogp_data["test_mark"]) == [False, True, True, True, True]

def test_mark_or_remove(clogp_data):
    # Test marking
    clogp_data["test_mark"] = [False, True, True, False, False]
    marked_df = mark_or_remove(clogp_data, mode="mark", mark_columns="test_mark")
    assert list(marked_df["test_mark"]) == [False, True, True, False, False]

    # Test removing
    removed_df = mark_or_remove(clogp_data, mode="remove", mark_columns="test_mark")
    assert "test_mark" not in removed_df.columns

    # Test invalid mode
    with pytest.raises(ValueError):
        mark_or_remove(clogp_data, mode="invalid_mode", mark_columns="test_mark")

def test_smarts_filter(test_data):
    # Test SMARTS filter
    smarts_filter = SMARTSFilter(
        smarts="C(=O)N",
        filter_name="amide_filter",
        filter_type="exclude"
    )
    filtered_df = smarts_filter.filter(test_data, mode="mark")
    assert len(filtered_df) == 0
