from pathlib import Path

from openadmet_toolkit.tests.datafiles import chembl_file, pubchem_file


def test_chembl_file():

    assert Path(chembl_file).exists()
    assert Path(pubchem_file).exists()