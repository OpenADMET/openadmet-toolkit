from pathlib import Path

from openadmet.toolkit.tests.datafiles import chembl_file, pubchem_file, filtering_file


def test_chembl_file():

    assert Path(chembl_file).exists()
    assert Path(pubchem_file).exists()
    assert Path(filtering_file).exists()
