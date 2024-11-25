"""
Unit and regression test for the openadmet_toolkit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest  # noqa: F401

import openadmet_toolkit  # noqa: F401


def test_openadmet_toolkit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openadmet_toolkit" in sys.modules
