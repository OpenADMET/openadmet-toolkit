from pathlib import Path
from openadmet_toolkit.tests.datafiles import example_file


def test_example_file():

    assert Path(example_file).exists()