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
    filt = BaseFilter.min_max_filter(
        property=clogp_data["clogp"],
        min_threshold=min_threshold,
        max_threshold=max_threshold,
    )
    assert list(filt) == [False, True, True, False, False]

    # Test with only max_threshold
    filt = BaseFilter.min_max_filter(
        property=clogp_data["clogp"],
        min_threshold=None,
        max_threshold=max_threshold,
    )
    assert list(filt) == [True, True, True, False, False]

    # Test with only min_threshold
    filt = BaseFilter.min_max_filter(
        property=clogp_data["clogp"],
        min_threshold=min_threshold,
        max_threshold=None,
    )
    assert list(filt) == [False, True, True, True, True]

def test_smarts_filter(test_data):
    smarts_df = pd.DataFrame({
        "smarts": ["Br"],
        "name": ["Bromine"],
        "smarts_column": ["Bromine"],
    })

    filter = SMARTSFilter(smarts_list=smarts_df['smarts'], names_list=smarts_df['name'])
    dat = filter.filter(smiles=test_data["cxsmiles"])
    assert dat.get_passes() == [False, False, True, True, False]

def test_datamol_filter(test_data):
    filter = DatamolFilter(
        name="clogp",
        min_value=-1.0,
        max_value=6.0,
        data_column="clogp",
    )
    dat = filter.filter(smiles=test_data["cxsmiles"])
    assert dat.get_passes() == [True, True, True, True, True]

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
    dat = filter.filter(smiles=test_data["cxsmiles"])
    assert dat.get_passes() == [False, False, False, True, False]


@pytest.mark.parametrize("min_dist", [1, None])
def test_proximity_filter_does_not_consume_its_own_threshold(min_dist):
    # The per-molecule measurement used to be written to self.min_dist, which is
    # the threshold filter() compares against, so the threshold became whichever
    # molecule was measured last and the verdicts depended on the input order.
    # Both of these have their O and Br between 1 and 20 bonds apart, 2 and 9
    # respectively, so both pass in either order, with or without a minimum.
    smiles = ["OCBr", "OCCCCCCCCBr"]

    # None means leaving min_dist at its default rather than passing it
    min_dist_kwargs = {} if min_dist is None else {"min_dist": min_dist}

    def run(order):
        filter = ProximityFilter(
            smarts_list_a=["O"],
            smarts_list_b=["Br"],
            names_list_a=["hydroxyl"],
            names_list_b=["bromine"],
            max_dist=20,
            **min_dist_kwargs,
        )
        output = filter.filter(smiles=pd.Series(order))
        return list(output.get_values()), output.get_passes(), filter.min_dist

    forward_dists, forward_passes, forward_threshold = run(smiles)
    reverse_dists, reverse_passes, reverse_threshold = run(smiles[::-1])

    assert forward_dists == [2, 9]
    assert reverse_dists == [9, 2]
    assert forward_passes == [True, True]
    assert reverse_passes == [True, True]
    assert forward_threshold == min_dist
    assert reverse_threshold == min_dist
