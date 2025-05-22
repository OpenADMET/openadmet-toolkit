import numpy as np
import pandas as pd
from openadmet.toolkit.chemoinformatics.rdkit_funcs import canonical_smiles
from medchem.catalogs import catalog_from_smarts

from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MolStandardize
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
tqdm.pandas()

import datamol as dm

from pydantic import BaseModel, Field
from openadmet.toolkit.filtering.filter_base import BaseFilter, min_max_filter, mark_or_remove

class SMARTSFilter(BaseFilter):
    """
    Filter class to filter a DataFrame based on SMARTS patterns.

    """
    smarts_df: pd.DataFrame = Field(description="DataFrame of SMARTS patterns to filter the smiles DataFrame.")
    smarts_column: str = Field(default="smarts", description="Column name in the DataFrame file containing SMARTS patterns.")
    names_column: str = Field(default="name", description="Column name in the DataFrame file containing names for the SMARTS patterns.")
    mark_column: str = Field(default="smarts_filtered", description="Column name to store the boolean marks (True/False).")

    def filter(self, df: pd.DataFrame, mode:str="mark", mol_col:str="mol") -> pd.DataFrame:
        """
        Run the SMARTS filter on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.
        mode : str
            Either "mark" or "remove". If "mark", the filter will mark the rows that meet the criteria
            either True or False. If "remove", the filter will remove the rows that meet the criteria.

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
        """
        # check if the smiles column exists
        if "canonical_smiles" not in df.columns:
            raise ValueError("The DataFrame must contain a 'canonical_smiles' column.")

        df["mol"] = df["canonical_smiles"].apply(
            lambda x: Chem.MolFromSmiles(x)
            )

        smarts_list = self.smarts_df[self.smarts_column].tolist()
        names_list = self.smarts_df[self.names_column].tolist()

        custom_catalog = catalog_from_smarts(smarts = smarts_list,
                                             labels = names_list,
                                             entry_as_inds = False,)

        df[self.mark_column] = df["mol"].apply(
            lambda x: custom_catalog.HasMatch(x)
        )

        return mark_or_remove(df, mode, self.mark_column)

class SMARTSProximityFilter(BaseFilter):
    """
    Filter class to filter two sites in a molecule based on their proximity.
    """
    def filter(self, df: pd.DataFrame, mode="mark") -> pd.DataFrame:
        """
        Run the SMARTS proximity filter on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.
        mode : str
            Either "mark" or "remove". If "mark", the filter will mark the rows that meet the criteria
            either True or False. If "remove", the filter will remove the rows that meet the criteria.

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
        """
        #TODO: figure out how to treat the sites

        return mark_or_remove(df, mode, "proximity")

    def get_match_min_dists(self, distances, chrom_inds, prot_ind):
        sub_dist_mat = distances[chrom_inds][:,prot_ind]
        return(sub_dist_mat.min())

    def get_min_dists(self, mol, chrom, prot_sites):
        distances = Chem.GetDistanceMatrix(mol)
        atom_matches_chrom, bond_matches_chrom = dm.substructure_matching_bonds(mol, chrom)
        min_dists = []
        if not atom_matches_chrom:
            return pd.NA
        for site in prot_sites:
            atom_matches_prot, bond_matches_prot = dm.substructure_matching_bonds(mol, site)
            if not atom_matches_prot:
                continue
            for prot_match in atom_matches_prot:
                for chrom_match in atom_matches_chrom:
                    min_dists.append(self.get_match_min_dists(distances, list(chrom_match), list(prot_match)))
        return(min_dists)

class pKaFilter(BaseFilter):
    """
    Filter class to filter a DataFrame based on pKa values.
    Parameters
    ----------
    min_pka : float
        The minimum pKa value for the range check (default is 3).
    max_pka : float
        The maximum pKa value for the range check (default is 11).
    min_unit_sep : float
        The minimum unit separation between pKa values (default is 1).
    """
    min_pka: float = Field(default=None, description="The minimum pKa value for the range check.")
    max_pka: float = Field(default=None, description="The maximum pKa value for the range check.")
    min_unit_sep: float = Field(default=None, description="The minimum unit separation between pKa values.")

    def filter(self, df: pd.DataFrame, pka_column: str = "pka", mode="mark") -> pd.DataFrame:
        """
        Run the pKa filter on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.
        mode : str
            Either "mark" or "remove". If "mark", the filter will mark the rows that meet the criteria
            either True or False. If "remove", the filter will remove the rows that meet the criteria.

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
        """
        # check if the pka column exists
        if pka_column not in df.columns:
            raise ValueError(f"The DataFrame does not contain a {pka_column} column.")

        if self.min_pka and self.max_pka:
            # filter for at least one pka between min_pka and max_pka
            df["in_range"] = df["pka"].apply(lambda x: self.pkas_valid_range(x))

        if self.min_unit_sep:
            # filter for pka values that are at least min_unit_sep apart
            df["unit_sep"] = df["pka"].apply(lambda x : self.pka_separation(x, self.min_unit_sep))

        return mark_or_remove(df, mode, ["in_range", "unit_sep"])

    def pkas_valid_range(self, pkas: list) -> bool:
        """
        Check if the pKa values are within the specified range.

        Parameters
        ----------
        pkas : list
            A list of pKa values to be checked.

        Returns
        -------
        bool
            True if all pKa values are within the specified range.
        """
        valid_range = False
        for pka in pkas:
            if self.min_pka <= pka <= self.max_pka:
                valid_range = True
                break
        return valid_range

    def pka_separation(pkas: list, min_unit_sep: float) -> bool:
        """
        Check if the pKa values are at least min_unit_sep apart.

        Parameters
        ----------
        pkas : list
            A list of pKa values to be checked.
        min_unit_sep : float
            The minimum unit separation between pKa values.

        Returns
        -------
        bool
            True if all pKa values are at least min_unit_sep apart.
        """
        for i in range(len(pkas)):
            for j in range(i + 1, len(pkas)):
                if abs(pkas[i] - pkas[j]) < 1:
                    return False
        return True

class logPFilter(BaseFilter):
    """
    Filter class to filter a DataFrame based on logP values.

    Parameters
    ----------
    min_logP : float
        The minimum logP value for the range check (default is 0).
    max_logP : float
        The maximum logP value for the range check (default is 5).
    """
    min_logP: float = Field(default=0, description="The minimum logP value for the range check.")
    max_logP: float = Field(default=5, description="The maximum logP value for the range check.")

    def filter(self, df: pd.DataFrame, mode="mark") -> pd.DataFrame:
        """
        Run the logP filter on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.  Must contain a 'logp' or 'clogp' column.
        mode : str
            Either "mark" or "remove". If "mark", the filter will mark the rows that meet the criteria
            either True or False. If "remove", the filter will remove the rows that meet the criteria.

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
        """
        if "logp" in df.columns:
            col = "logp"
        elif "clogp" in df.columns:
            col = "clogp"
        else:
            raise ValueError("The DataFrame must contain a 'logp' or 'clogp' column.")

        # filter for logP values between min_logP and max_logP
        df = min_max_filter(df, col, self.min_logP, self.max_logP, "logp_filtered")

        return mark_or_remove(df, mode, "logp_filtered")
