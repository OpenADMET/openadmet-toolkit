import pandas as pd
import pytest
from openff.units import unit

from openadmet.toolkit.chemoinformatics.rdkit_funcs import (
    canonical_smiles,
    old_standardize_smiles,
    run_reaction,
    smiles_to_inchikey,
)

from openadmet.toolkit.chemoinformatics.activity_funcs import (
    calculate_pac50,
    pac50_to_ki,
    ki_to_dg
)

from openadmet.toolkit.chemoinformatics.retrosynth import ReactionSMART, Retrosynth


@pytest.fixture()
def smi():
    return "c1ccccc1"


@pytest.fixture()
def amide_smarts():
    return "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OX2H1].[N:3]"


@pytest.fixture()
def amide_dataframe():
    data = pd.DataFrame(
        {
            "SMILES": ["CCN(CC)CC(=O)NC1=C(C=CC=C1C)C"],
            "INCHIKEY": ["NNJVILVZKWQKPM-UHFFFAOYSA-N"],
            "common_name": ["lidocaine"],
        }
    )
    return data


# simple tests
def test_standardize_smiles_old(smi):
    assert old_standardize_smiles(smi)


def test_canonical_smiles(smi):
    assert canonical_smiles(smi)


def test_smi_to_inchikey(smi):
    assert smiles_to_inchikey(smi)


def test_run_reaction(amide_smarts):
    # meloxicam
    res = run_reaction("CC1=CN=C(S1)NC(=O)C2=C(C3=CC=CC=C3S(=O)(=O)N2C)O", amide_smarts)
    assert len(res) == 2
    assert res[0] == "CN1C(C(=O)O)=C(O)c2ccccc2S1(=O)=O"
    assert res[1] == "Cc1cnc(N)s1"


def test_reaction_smart_retro(amide_dataframe):
    rs = ReactionSMART(
        reaction="[C:1](=[O:2])[N:3]>>[C:1](=[O:2])[OX2H1].[N:3]",
        reaction_name="amide_coupling",
        product_names=["carboxylic_acid", "amine"],
        reactant_names=["amide"],
    )
    retro = Retrosynth(reaction=rs)
    df = retro.run(amide_dataframe, smiles_column="SMILES")
    assert all(pd.notna(df["amide_coupling_amine_inchikey"]))
    assert all(pd.notna(df["amide_coupling_carboxylic_acid_inchikey"]))

def test_calculate_pac50():
    ic50_1 = calculate_pac50(1, input_unit_str="uM")
    ic50_2 = calculate_pac50(10, input_unit_str="nM")
    ic50_3 = calculate_pac50(3e-8, input_unit_str="M")
    assert ic50_1 == 6.0
    assert ic50_2 == 8.0
    assert round(ic50_3, 1) == 7.5

def test_pac50_to_ki():
    ki = pac50_to_ki(9.0)
    assert ki == 1e-9 * unit.molar

def test_ki_to_dg():
    dg = ki_to_dg(ki=100, input_unit_str="nM")
    assert round(dg, 1) == -40.0 * unit.kilojoule_per_mole
