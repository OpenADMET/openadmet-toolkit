import pytest
from pydantic import ValidationError

from openadmet.toolkit.database import chembl
from openadmet.toolkit.database.chembl import (
    Caco2ChEMBLCurator,
    ChEMBLDatabaseConnector,
    PermissiveChEMBLTargetCurator,
)

LATEST = 36


@pytest.fixture()
def requested_versions(monkeypatch):
    """Stub out the ChEMBL download, recording the versions requested."""
    versions = []

    def _download(version=None, **kwargs):
        versions.append(version)
        return "chembl.db"

    monkeypatch.setattr(chembl.chembl_downloader, "download_extract_sqlite", _download)
    return versions


@pytest.fixture()
def latest_calls(monkeypatch):
    """Stub out the network lookup of the latest ChEMBL version, counting calls."""
    calls = []

    def _latest(**kwargs):
        calls.append(True)
        return str(LATEST)

    monkeypatch.setattr(chembl.chembl_downloader, "latest", _latest)
    return calls


def test_default_chembl_version_is_latest(requested_versions, latest_calls):
    curator = Caco2ChEMBLCurator()
    # the "latest" default is resolved and backfilled onto the curator
    assert curator.chembl_version == LATEST
    assert curator._chembl_connector.version == LATEST
    assert len(latest_calls) == 1
    assert requested_versions == [str(LATEST)]


@pytest.mark.parametrize("kwarg", ["chembl_version", "version"])
def test_chembl_version_kwargs(requested_versions, latest_calls, kwarg):
    curator = PermissiveChEMBLTargetCurator(
        chembl_target_id="CHEMBL3356", standard_type="IC50", **{kwarg: 35}
    )
    # an explicit version takes precedence and never triggers the latest lookup
    assert curator.chembl_version == 35
    assert latest_calls == []
    assert requested_versions == ["35"]


def test_conflicting_version_kwargs_raise(requested_versions, latest_calls):
    with pytest.raises(ValidationError):
        Caco2ChEMBLCurator(chembl_version=35, version=36)
    assert requested_versions == []
    assert latest_calls == []


def test_unknown_kwargs_raise(requested_versions, latest_calls):
    with pytest.raises(ValidationError, match="chembl_ver"):
        Caco2ChEMBLCurator(chembl_ver=35)
    assert requested_versions == []
    assert latest_calls == []


def test_latest_lookup_failure_gives_clear_error(requested_versions, monkeypatch):
    def _latest(**kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(chembl.chembl_downloader, "latest", _latest)
    with pytest.raises(RuntimeError, match="explicit `chembl_version`"):
        Caco2ChEMBLCurator()
    assert requested_versions == []


def test_connector_create_with_explicit_version(requested_versions, latest_calls):
    connector = ChEMBLDatabaseConnector.create_chembl_database(version=34)
    assert connector.version == 34
    assert latest_calls == []
    assert requested_versions == ["34"]
