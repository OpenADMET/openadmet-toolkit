import pytest

from openadmet.toolkit.chemoinformatics.multi_data_curation import MultiDataProcessing
from openadmet.toolkit.tests.datafiles import chembl_data_yaml, processed_chembl_data_yaml, pubchem_data_yaml, processed_pubchem_data_yaml


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

@pytest.mark.parametrize(
        "yaml, pchembl, expected_cols",
        [
            (chembl_data_yaml, True, ["OPENADMET_INCHIKEY", "OPENADMET_CANONICAL_SMILES", "OPENADMET_LOGAC50", "OPENADMET_ACTIVITY_TYPE"]),
            (pubchem_data_yaml, False, ["OPENADMET_INCHIKEY", "OPENADMET_CANONICAL_SMILES", "OPENADMET_LOGAC50", "OPENADMET_ACTIVITY_TYPE"])
        ]
)
def test_batch_process(yaml, pchembl, expected_cols):
    data = MultiDataProcessing.batch_process(path=yaml, pchembl=pchembl, savefile=False)

    # For every dataframe processed, check that the required columns are there
    for target_name, df in data.items():
        missing = [col for col in expected_cols if col not in df.columns]
        assert not missing, f"Missing columns in target '{target_name}': {missing}."

@pytest.mark.parametrize(
    "yaml, process, pchembl, expected_length",
    [
        (processed_chembl_data_yaml, False, True, 22280),
        (chembl_data_yaml, True, True, 22280),
        (processed_pubchem_data_yaml, False, False, 8179),
        (pubchem_data_yaml, True, False, 8179)
    ]
)
def test_multitask_process(yaml, process, pchembl, expected_length):
    data = MultiDataProcessing.multitask_process(path=yaml, process=process, pchembl=pchembl, savemultifile=False)

    assert len(data) == expected_length
