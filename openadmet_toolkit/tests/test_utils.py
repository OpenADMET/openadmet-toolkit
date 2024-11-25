import pytest
import pandas as pd
from openadmet_toolkit.utils.presentation import make_pptx_from_molecule_data


@pytest.fixture
def molecule_dataframe():
    N=10
    smiles = ["C"*i for i in range(N)]
    data1 = [f"A{i}" for i in range(N)]
    data2 = [f"B{i}" for i in range(N)]
    df = pd.DataFrame({"SMILES1": smiles, "DATA1":data1, "DATA2":data2})
    return df


def test_make_pptx_from_molecule_data(molecule_dataframe, tmp_path):
    make_pptx_from_molecule_data(molecule_dataframe, tmp_path/"test.pptx", smiles_col="SMILES1", legend_columns=["DATA1", "DATA2"])