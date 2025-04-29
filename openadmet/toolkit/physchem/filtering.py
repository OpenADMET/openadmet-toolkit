import numpy as np
import pandas as pd
from openadmet.toolkit.chemoinformatics.rdkit_funcs import canonical_smiles
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolStandardize
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas()

import datamol as dm

def filter_by_property(csv_path, min_pka=3, max_pka=11, min_unit_sep=1):
    process_catalog(csv_path)

def basic_filter(df, property, min_threshold, max_threshold):
    """
    Filters a DataFrame based on a property value range.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to filter.
    property : str
        The name of the column in the DataFrame representing the property to filter on.
    min_threshold : float
        The minimum value of the property.
    max_threshold : float
        The maximum value of the property.

    Returns
    -------
    pandas.DataFrame
        A filtered DataFrame containing only rows where the property values are within the
        specified range.

    """
    return df[(df[property] >= min_threshold) & (df[property] <= max_threshold)]

def process_catalog(path):
    """
    Processes a chemical catalog file to remove na values and compute molecular descriptors.

    Parameters
    ----------
    path : str
        The file path to the CSV file containing the chemical catalog.
        The file must include a 'smiles' column with SMILES strings.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the original data along with computed molecular descriptors.
        The DataFrame includes the following additional columns:
        - 'canonical_smiles': Canonicalized SMILES strings.
        - 'mol': RDKit molecule objects.
        - Molecular descriptors computed for each molecule.

    """
    df = pd.read_csv(path)
    df = df.dropna(subset=['smiles'])
    with dm.without_rdkit_log():
        df['canonical_smiles'] = df['smiles'].progress_apply(lambda x: canonical_smiles(x))
        df = df.dropna(subset=['canonical_smiles'])
    df["mol"] = df["canonical_smiles"].progress_apply(dm.to_mol)
    df = df.dropna(subset="mol")
    df_desc = dm.descriptors.batch_compute_many_descriptors(df["mol"], progress=True)
    return(pd.concat([df.reset_index(), df_desc.reset_index()], axis=1))

def filter_pkas(pkas, min_pka=3, max_pka=11, min_unit_sep=1):
    """
    Filters a list of pKa values based on specific criteria.

    Parameters
    ----------
    pkas : list[float]
        A list of pKa values to be filtered.
    min_pka : float
        The minimum pKa value for the range check (default is 3).
    max_pka : float
        The maximum pKa value for the range check (default is 11).
    min_unit_sep : float
        The minimum distance between pKa values (default is 1).
    Returns
    -------
    bool
        True if the list of pKa values passes all checks, False otherwise.
    """

    # Check if list is empty
    if len(pkas) == 0:
        return False

    # Check if at AT LEAST ONE pKa is between min_pka and max_pka (inclusive)
    valid_range = False
    for pka in pkas:
        if min_pka <= pka <= max_pka:
            valid_range = True
            break

    if not valid_range:
        return False

    # Check if all pKa values are at least unit_sep pka units apart from each other
    for i in range(len(pkas)):
        for j in range(i + 1, len(pkas)):
            if abs(pkas[i] - pkas[j]) < min_unit_sep:
                return False

    # If we've passed all checks, return True
    return True


def get_match_min_dists(distances, chrom_inds, prot_ind):
    """
    Computes the minimum distance between chromophore atoms and a protonation site.

    Parameters
    ----------
    distances : numpy.ndarray
        A 2D distance matrix representing the pairwise distances between atoms in the molecule.
    chrom_inds : list[int]
        A list of indices corresponding to the atoms in the chromophore.
    prot_ind : int
        The index of the atom representing the protonation site.

    Returns
    -------
    float
        The minimum distance between the chromophore atoms and the protonation site.
    """
    sub_dist_mat = distances[chrom_inds][:,prot_ind]
    return(sub_dist_mat.min())


def get_min_dists_multi(mol, chromophores, prot_sites):
    """
    Computes the minimum distances between chromophore atoms and protonation sites in a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        The RDKit molecule object for which distances are computed.
    chromophores : list[rdkit.Chem.Mol]
        A list of RDKit molecule objects representing chromophore substructures.
    prot_sites : list[rdkit.Chem.Mol]
        A list of RDKit molecule objects representing protonation site substructures.

    Returns
    -------
    list[float]
        A list of minimum distances between chromophore atoms and protonation sites.
        Each entry corresponds to the minimum distance for a specific chromophore-protonation site pair.

    """
    distances = Chem.GetDistanceMatrix(mol)
    min_dists = []
    for chrom in chromophores:
        atom_matches_chrom, bond_matches_chrom = dm.substructure_matching_bonds(mol, chrom)
        if not atom_matches_chrom:
            continue
        for site in prot_sites:
            atom_matches_prot, bond_matches_prot = dm.substructure_matching_bonds(mol, site)
            if not atom_matches_prot:
                continue
            for prot_match in atom_matches_prot:
                for chrom_match in atom_matches_chrom:
                    min_dists.append(get_match_min_dists(distances, list(chrom_match), list(prot_match)))
    return(min_dists)
