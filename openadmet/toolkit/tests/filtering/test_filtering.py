import pandas as pd
import pytest

from openadmet.toolkit.filtering.filter_base import BaseFilter

from openadmet.toolkit.filtering.physchem_filters import pKaFilter
from openadmet.toolkit.filtering.physchem_filters import DatamolFilter
from openadmet.toolkit.filtering.physchem_filters import SMARTSFilter
from openadmet.toolkit.filtering.physchem_filters import ProximityFilter

from openadmet.toolkit.tests.datafiles import filtering_file

@pytest.fixture()
def clogp_data():
    df = pd.DataFrame({
        "clogp": [-2.0, 1.5, 3.0, 7.0, 8.0],
        "test_mark": [False, True, True, False, False]
    })
    return df

@pytest.fixture()
def pka_data():
    df = pd.DataFrame({
        "pka": [[3.0, 5.0], [7.0], [9.0, 11.0], [2.0], [11.0, 11.5]],
        "test_mark": [False] * 5
    })
    return df

@pytest.fixture()
def test_data():
    filtering_df = pd.read_csv(filtering_file)
    return filtering_df

def test_min_max_filter(clogp_data):
    min_threshold = -1.0
    max_threshold = 6.0

    # Test with both min_threshold and max_threshold
    BaseFilter.min_max_filter(
        df=clogp_data,
        property="clogp",
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        mark_column="test_mark"
    )
    assert list(clogp_data["test_mark"]) == [False, True, True, False, False]

    # Test with only max_threshold
    BaseFilter.min_max_filter(
        df=clogp_data,
        property="clogp",
        min_threshold=None,
        max_threshold=max_threshold,
        mark_column="test_mark"
    )
    assert list(clogp_data["test_mark"]) == [True, True, True, False, False]

    # Test with only min_threshold
    BaseFilter.min_max_filter(
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
    marked_df = BaseFilter.mark_or_remove(clogp_data, mode="mark", mark_columns="test_mark")
    assert list(marked_df["test_mark"]) == [False, True, True, False, False]

    # Test removing
    removed_df = BaseFilter.mark_or_remove(clogp_data, mode="remove", mark_columns="test_mark")
    assert "test_mark" not in removed_df.columns
    assert len(removed_df) == 2  # Only rows with True in test_mark should remain

    # Test invalid mode
    with pytest.raises(ValueError):
        BaseFilter.mark_or_remove(clogp_data, mode="invalid_mode", mark_columns="test_mark")

def test_smarts_filter(test_data):
    smarts_df = pd.DataFrame({
        "smarts": ["Br"],
        "name": ["Bromine"],
        "smarts_column": ["Bromine"],
    })

    filter = SMARTSFilter(smarts_list=smarts_df['smarts'], names_list=smarts_df['name'], names_column='bromine')
    filtered_df = filter.filter(df=test_data, smiles_column="cxsmiles", mode="mark")
    assert list(filtered_df["passed_smarts_filter"]) == [False, False, True, True, False]

def test_datamol_filter(test_data):
    filter = DatamolFilter(
        name="clogp",
        min_value=-1.0,
        max_value=6.0,
        data_column="clogp",
    )
    filtered_df = filter.filter(test_data, mode="mark", smiles_column="cxsmiles")
    assert list(filtered_df["passed_clogp_filter"]) == [True, True, True, True, True]

def test_proximity_filter(test_data):
    filter = ProximityFilter(
        smarts_list_a = ["O"],
        smarts_list_b = ["Br"],
        names_list_a = ["alkyl"],
        names_list_b = ["bromine"],
        smarts_column_a = "alkyl_smarts",
        smarts_column_b = "bromine_smarts",
        max_dist=6.0,
    )
    filtered_df = filter.filter(test_data, mode="mark", smiles_column="cxsmiles")
    assert list(filtered_df["passed_proximity_filter"]) == [False, False, False, True, False]
