import pytest
from pydantic import ValidationError

from openadmet.toolkit.database import chembl
from openadmet.toolkit.database.chembl import (
    Caco2ChEMBLCurator,
    ChEMBLDatabaseConnector,
    PermissiveChEMBLTargetCurator,
)


@pytest.fixture()
def requested_versions(monkeypatch):
    """Stub out the ChEMBL download, recording the versions requested."""
    versions = []

    def _create(version):
        versions.append(version)
        return ChEMBLDatabaseConnector(version=version, sqlite_path="chembl.db")

    monkeypatch.setattr(
        chembl.ChEMBLDatabaseConnector, "create_chembl_database", staticmethod(_create)
    )
    return versions


def test_default_chembl_version(requested_versions):
    curator = Caco2ChEMBLCurator()
    assert curator.chembl_version == 34
    assert requested_versions == [34]


@pytest.mark.parametrize("kwarg", ["chembl_version", "version"])
def test_chembl_version_kwargs(requested_versions, kwarg):
    curator = PermissiveChEMBLTargetCurator(
        chembl_target_id="CHEMBL3356", standard_type="IC50", **{kwarg: 35}
    )
    assert curator.chembl_version == 35
    assert requested_versions == [35]


def test_conflicting_version_kwargs_raise(requested_versions):
    with pytest.raises(ValidationError):
        Caco2ChEMBLCurator(chembl_version=35, version=36)
    assert requested_versions == []


def test_unknown_kwargs_raise(requested_versions):
    with pytest.raises(ValidationError, match="chembl_ver"):
        Caco2ChEMBLCurator(chembl_ver=35)
    assert requested_versions == []
