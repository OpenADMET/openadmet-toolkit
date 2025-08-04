from importlib import resources

import openadmet.toolkit.tests.test_data  # noqa: F401

_data_ref = resources.files("openadmet.toolkit.tests.test_data")  # noqa: F841


chembl_file = (_data_ref / "chembl_1A2_test_data.csv").as_posix()
pubchem_file = (_data_ref / "AID_410_pubchem_test_data.csv").as_posix()
filtering_file = (_data_ref / "filtering_test_data.csv").as_posix()

# data yamls
chembl_data_yaml = (_data_ref / "pchembl_data.yaml").as_posix()
processed_chembl_data_yaml = (_data_ref / "processed_pchembl_data.yaml").as_posix()
pubchem_data_yaml = (_data_ref / "pubchem_data.yaml").as_posix()
processed_pubchem_data_yaml = (_data_ref / "processed_pubchem_data.yaml").as_posix()