from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd


def standardize_smiles(smiles: str, raise_error:bool=False) -> str:
    """
    Standardize a SMILES string to a canonical form. Taken from Pat Walters method.
    If the SMILES string cannot be standardized, it will return a pd.NA.

    Parameters
    ----------
    smiles : str
        SMILES string to standardize.
    raise_error : bool, optional
        Raise an error if the SMILES string cannot be standardized, by default False
    
    """
    try:
        # follows the steps in
        # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
        # as described **excellently** (by Greg) in
        # https://www.youtube.com/watch?v=eWTApNX8dJQ
        mol = Chem.MolFromSmiles(smiles)
    
        # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
        clean_mol = rdMolStandardize.Cleanup(mol)
    
        # if many fragments, get the "parent" (the actual mol we are interested in) 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
    
        # try to neutralize molecule
        uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    
        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.
    
        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    
        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
    except:
        if raise_error:
            raise ValueError(f"Could not standardize SMILES: {smiles}")
        else:
            return pd.NA
        
def smiles_to_inchikey(smiles: str, raise_error:bool=False) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol)
    except:
        if raise_error:
            raise ValueError(f"Could not convert SMILES to InChIKey: {smiles}")
        else:
            return pd.NA