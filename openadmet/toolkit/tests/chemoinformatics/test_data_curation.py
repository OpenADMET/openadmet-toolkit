import pandas as pd
import pytest

from openadmet.toolkit.chemoinformatics.data_curation import ChEMBLProcessing, PubChemProcessing, DataProcessing, MultiDataProcessing
from openadmet.toolkit.tests.datafiles import chembl_file, pubchem_file, chembl_data_yaml, processed_chembl_data_yaml, pubchem_data_yaml, processed_pubchem_data_yaml


@pytest.fixture()
def chembl():
    return chembl_file


@pytest.fixture()
def pubchem():
    return pubchem_file

@pytest.fixture()
def chembl_data():
    return chembl_data_yaml

@pytest.fixture()
def pubchem_data():
    return pubchem_data_yaml

@pytest.fixture()
def processed_chembl_data():
    return processed_chembl_data_yaml

@pytest.fixture()
def processed_pubchem_data():
    return processed_pubchem_data_yaml


def test_chembl_inhib():
    chembl_inhib = ChEMBLProcessing(inhib=True)
    df = chembl_inhib.process(chembl_file)
    assert all(pd.notna(df["Smiles"]))
    assert all(pd.notna(df["OPENADMET_CANONICAL_SMILES"]))
    assert df["OPENADMET_INCHIKEY"].is_unique


def test_chembl_react():
    chembl_react = ChEMBLProcessing(react=True)
    df = chembl_react.process(chembl_file)
    assert all(pd.notna(df["Smiles"]))
    assert all(pd.notna(df["OPENADMET_CANONICAL_SMILES"]))
    assert df["OPENADMET_INCHIKEY"].is_unique


def test_pubchem_inhib():
    pubchem_inhib = PubChemProcessing(inhib=True)
    df = pubchem_inhib.process(pubchem_file, "test1", "test2")
    assert all(pd.notna(df["Smiles"]))
    assert all(pd.notna(df["OPENADMET_CANONICAL_SMILES"]))
    assert df["OPENADMET_INCHIKEY"].is_unique


def test_pubchem_react():
    pubchem_inhib = PubChemProcessing(react=True)
    df = pubchem_inhib.process(pubchem_file, "test1", "test2")
    assert all(pd.notna(df["Smiles"]))
    assert all(pd.notna(df["OPENADMET_CANONICAL_SMILES"]))
    assert df["OPENADMET_INCHIKEY"].is_unique

def test_single_processing():
    data_process = DataProcessing()
    df = data_process.read_file(pubchem_file)
    df = data_process.standardize_smiles_and_convert(df, smiles_col="PUBCHEM_EXT_DATASOURCE_SMILES")
    assert all(pd.notna(df["OPENADMET_CANONICAL_SMILES"]))
    assert "OPENADMET_INCHIKEY" in df.columns

@pytest.mark.parametrize(
        "yaml, log_transform, expected_cols",
        [
            (chembl_data_yaml, True, ["OPENADMET_INCHIKEY", "OPENADMET_CANONICAL_SMILES", "OPENADMET_LOGAC50", "OPENADMET_ACTIVITY_TYPE"]),
            (pubchem_data_yaml, False, ["OPENADMET_INCHIKEY", "OPENADMET_CANONICAL_SMILES", "OPENADMET_LOGAC50", "OPENADMET_ACTIVITY_TYPE"])
        ]
)
def test_batch_process(yaml, log_transform, expected_cols):
    data = MultiDataProcessing.batch_process(path=yaml, log_transform=log_transform, savefile=False)

    # For every dataframe processed, check that the required columns are there
    for target_name, df in data.items():
        missing = [col for col in expected_cols if col not in df.columns]
        assert not missing, f"Missing columns in target '{target_name}': {missing}."

@pytest.mark.parametrize(
    "yaml, process, log_transform, expected_length",
    [
        (processed_chembl_data_yaml, False, True, 22290),
        (chembl_data_yaml, True, True, 22290),
        (processed_pubchem_data_yaml, False, False, 8179),
        (pubchem_data_yaml, True, False, 8179)
    ]
)
def test_multitask_process(yaml, process, log_transform, expected_length):
    data = MultiDataProcessing.multitask_process(path=yaml, process=process, log_transform=log_transform, savemultifile=False)

    assert len(data) == expected_length
