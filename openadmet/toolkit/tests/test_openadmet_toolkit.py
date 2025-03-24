"""
Unit and regression test for the openadmet.toolkit package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest  # noqa: F401

import openadmet.toolkit  # noqa: F401


def test_openadmet.toolkit_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "openadmet.toolkit" in sys.modules
