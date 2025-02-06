from importlib import resources

import openadmet_toolkit.tests.test_data  # noqa: F401

_data_ref = resources.files("openadmet_toolkit.tests.test_data")  # noqa: F841


chembl_file = (_data_ref / "chembl_1A2_test_data.csv").as_posix()
pubchem_file = (_data_ref / "AID_410_pubchem_test_data.csv").as_posix()
