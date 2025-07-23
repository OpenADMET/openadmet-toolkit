import pandas as pd
import pytest

from openadmet.toolkit.chemoinformatics.data_curation import ChEMBLProcessing, PubChemProcessing, DataProcessing
from openadmet.toolkit.tests.datafiles import chembl_file, pubchem_file


@pytest.fixture()
def chembl():
    return chembl_file


@pytest.fixture()
def pubchem():
    return pubchem_file


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
