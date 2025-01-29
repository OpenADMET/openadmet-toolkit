from importlib import resources

import openadmet_toolkit.tests.test_data  # noqa: F401

_data_ref = resources.files("openadmet_toolkit.tests.test_data") # noqa: F841


# example_file = (_data_ref / "example_file.txt").as_posix()
