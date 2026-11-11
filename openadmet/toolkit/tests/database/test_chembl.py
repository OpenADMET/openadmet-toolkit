import pandas as pd
import pytest
from pydantic import ValidationError

from openadmet.toolkit.database import chembl
from openadmet.toolkit.database.chembl import (
    PAPP_UNIT_FACTORS,
    Caco2ChEMBLCurator,
    ChEMBLDatabaseConnector,
    MDCKChEMBLCurator,
    PermissiveChEMBLTargetCurator,
    papp_to_micro_cm_per_s,
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


def _admitted_units(query):
    """The unit strings a Papp query's `standard_units` filter lets through."""
    line = next(ln for ln in query.splitlines() if "activities.standard_units in" in ln)
    literals = line[line.index("(") + 1:line.rindex(")")].split(",")
    # strip the surrounding quotes and undo SQL's doubled-quote escaping
    return sorted(lit.strip().strip("'").replace("''", "'") for lit in literals)


@pytest.mark.parametrize(
    "curator_factory",
    [
        lambda: Caco2ChEMBLCurator(),
        lambda: MDCKChEMBLCurator(cell_type="MDCK-II"),
    ],
    ids=["caco2", "mdck"],
)
def test_papp_queries_only_admit_convertible_units(
    requested_versions, latest_calls, curator_factory
):
    # Every unit a Papp query can return has to be convertible onto the unit the
    # curator relabels it with. A unit that gets through the filter but is not in
    # the table would be left unscaled and then labeled as if it had been scaled.
    admitted = _admitted_units(curator_factory().get_templated_query())
    assert admitted
    assert all(unit.lower() in PAPP_UNIT_FACTORS for unit in admitted)


@pytest.mark.parametrize(
    "unit,factor",
    [
        ("ucm/s", 1.0),
        ("10^-6 cm/s", 1.0),
        ("10-6 cm/s", 1.0),
        ("10'-6 cm/s", 1.0),
        ("10^-5 cm/s", 10.0),
        ("10-5 cm/s", 10.0),
        ("10'-5 cm/s", 10.0),
        ("nm/s", 0.1),
    ],
)
def test_papp_units_are_scaled_before_relabeling(unit, factor):
    df = pd.DataFrame({"standard_value": [2.0], "standard_units": [unit]})
    out = papp_to_micro_cm_per_s(df)
    assert out["standard_value"].tolist() == [2.0 * factor]
    assert out["standard_units"].tolist() == ["10^-6 cm/s"]


def test_unconvertible_papp_unit_raises_rather_than_being_relabeled():
    # "cm/s" is six orders of magnitude from the label the curator applies, so
    # passing it through unscaled would make the value unrecoverably wrong.
    df = pd.DataFrame({"standard_value": [1.0], "standard_units": ["cm/s"]})
    with pytest.raises(ValueError, match="cm/s"):
        papp_to_micro_cm_per_s(df)


@pytest.mark.parametrize(
    "curator_factory",
    [
        lambda: Caco2ChEMBLCurator(),
        lambda: MDCKChEMBLCurator(cell_type="MDCK-II"),
    ],
    ids=["caco2", "mdck"],
)
def test_papp_curators_convert_the_data_they_return(
    requested_versions, latest_calls, monkeypatch, curator_factory
):
    raw = pd.DataFrame(
        {
            "standard_value": [1.0, 1.0],
            "standard_units": ["10^-6 cm/s", "10'-5 cm/s"],
        }
    )
    monkeypatch.setattr(
        chembl.ChEMBLCuratorBase, "get_activity_data", lambda self, return_as="df": raw.copy()
    )
    df = curator_factory().get_activity_data()
    assert df["standard_value"].tolist() == [1.0, 10.0]
    assert df["standard_units"].tolist() == ["10^-6 cm/s"] * 2


def test_mdck_cell_type_survives_validation(requested_versions, latest_calls):
    # The validator used to drop the value it had just checked, which left
    # cell_type as None and produced a query matching the literal string "None".
    curator = MDCKChEMBLCurator(cell_type="MDCK-MDR1")
    assert curator.cell_type == "MDCK-MDR1"
    assert "assays.assay_cell_type = 'MDCK-MDR1'" in curator.get_templated_query()


def test_mdck_rejects_an_unknown_cell_type(requested_versions, latest_calls):
    with pytest.raises(ValidationError, match="Invalid cell type"):
        MDCKChEMBLCurator(cell_type="Caco-2")
