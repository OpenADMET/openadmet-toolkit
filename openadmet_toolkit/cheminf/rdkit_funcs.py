import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize





def smiles_to_inchikey(smiles: str, raise_error: bool = False) -> str:
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchiKey(mol)
    except Exception as e:
        if raise_error:
            raise ValueError(
                f"Could not convert SMILES to InChIKey: {smiles} with error: {e}"
            )
        else:
            return pd.NA


def run_reaction(
    smiles: str, reaction_smarts: str, return_as="smiles", raise_error: bool = False
):
    try:
        mol = Chem.MolFromSmiles(smiles)
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        products = reaction.RunReactants((mol,))

        for product_set in products:
            frags = []
            for fragment in product_set:
                frags.append(fragment)

            if return_as == "smiles":
                return [Chem.MolToSmiles(frag) for frag in frags]
            else:
                return frags

        else:
            return pd.NA
    except Exception as e:
        if raise_error:
            raise ValueError(
                f"Could not run reaction: {reaction_smarts} on SMILES: {smiles} with error: {e}"
            )
        else:
            return pd.NA



def canonical_smiles(smiles: str, raise_error: bool = False, remove_salt: bool=True) -> str:
    """
    Standardizes a SMILES string by removing salts, normalizing the molecule, 
    finding its canonical tuatomer, and returning the kekulized canonical SMILES.

    Parameters
    ----------
    smiles : str
        The SMILES string to standardize.
    raise_error : bool, optional
        Raise an error if the SMILES string cannot be standardized, by default False
    remove_salt : bool, optional
        Remove salts from the molecule, by default True
    
    Returns
    -------
    str
    """
    try:
        # Step 1: Convert the SMILES string to an RDKit molecule object.
        mol0 = Chem.MolFromSmiles(smiles)

        # Step 2: Remove salts from the molecule.
        if remove_salt:
            remover = Chem.SaltRemover.SaltRemover()
            mol1 = remover.StripMol(mol0)
        else:
            mol1 = mol0

        # Step 3: Remove hydrogens, disconnect metals, normalize the molecule, and reionize.
        mol2_cleaned = rdMolStandardize.Cleanup(mol1)

        # Step 4: If the molecule has multiple fragments, get the "parent" fragment.
        mol3_parent = rdMolStandardize.FragmentParent(mol2_cleaned)

        # Step 5: Neutralize the molecule.
        uncharger = rdMolStandardize.Uncharger()
        mol4_neut = uncharger.uncharge(mol3_parent)

        # Step 6: Find the canonical tautomer of the molecule.
        enumerator = rdMolStandardize.TautomerEnumerator()
        enumerator.SetRemoveBondStereo(False)  # Keep bond stereochemistry
        enumerator.SetRemoveSp3Stereo(False)  # Keep tetrahedral stereochemistry
        mol5_cantaut = enumerator.Canonicalize(mol4_neut)

        # Step 7: Get the canonical SMILES representation of the standardized molecule.
        # The molecule object must be converted to SMILES and back, to standardize dearomatization.
        smiles1_no_kekule = Chem.MolToSmiles(mol5_cantaut)
        mol6_no_kekule = Chem.MolFromSmiles(smiles1_no_kekule)
        smiles2_oa_canonical = Chem.MolToSmiles(mol6_no_kekule, canonical=True, kekuleSmiles=True)

        return smiles2_oa_canonical

    except Exception as e:
        if raise_error:
            raise ValueError(f"Could not standardize SMILES: {smiles} with error: {e}")
        else:
            return pd.NA

def old_standardize_smiles(smiles: str, raise_error: bool = False) -> str:
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
        uncharger = (
            rdMolStandardize.Uncharger()
        )  # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

        # note that no attempt is made at reionization at this step
        # nor at ionization at some pH (rdkit has no pKa caculator)
        # the main aim to represent all molecules from different sources
        # in a (single) standard way, for use in ML, catalogue, etc.

        te = rdMolStandardize.TautomerEnumerator()  # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)

        return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)
    except Exception as e:
        if raise_error:
            raise ValueError(f"Could not standardize SMILES: {smiles} with error: {e}")
        else:
            return pd.NA