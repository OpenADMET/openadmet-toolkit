import pandas as pd
import pytest

from openadmet_toolkit.cheminf.rdkit_funcs import (run_reaction,
                                                   smiles_to_inchikey,
                                                   standardize_smiles)
from openadmet_toolkit.cheminf.retrosynth import ReactionSMART, Retrosynth


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


def test_standardize_smiles(smi):
    assert standardize_smiles(smi)


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
