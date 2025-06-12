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
from openadmet.toolkit.filtering.filter_base import BaseFilter
from typing import ClassVar

# - ProximityFilter and add test for it

class SMARTSFilter(BaseFilter):
    """
    Filter class to filter a DataFrame based on SMARTS patterns.

    """
    smarts_list: list
    names_list: list
    include: bool = True # whether to filter for or against smarts

    names_column:str
    mark_column: ClassVar[str] = "passed_smarts_filter"

    def calculate(self,
                  df: pd.DataFrame,
                  smiles_column: str = "OPENADMET_CANONICAL_SMILES",
                  mol_column: str = "mol") -> pd.DataFrame:
        """
        Run the SMARTS filter on the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.
        smarts_list : list
            The list of SMARTS patterns to match in the DataFrame.
        names_list : list
            The list of names corresponding to the SMARTS patterns.
        smarts_col : str
            The column name to write to containing .
        mol_col : str
            The column name containing the RDKit Mol objects (default is 'mol').

        Returns
        -------
        pandas.DataFrame
            The DataFrame with .
        """

        df = self.set_mol_column(df=df, smiles_column=smiles_column, mol_column=mol_column)

        df[self.names_column] = df[mol_column].apply(
            lambda x: self.match_smarts_by_catalog(x, self.smarts_list, self.names_list)
        )

        return df

    def filter(self,
               df: pd.DataFrame,
               smiles_column: str = "OPENADMET_CANONICAL_SMILES",
               mol_column: str = "mol",
               mode: str = "mark",
               calculate: bool = "True") -> pd.DataFrame:
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
        
        if calculate:
            df = self.calculate(df, smiles_column, mol_column)

        df = self.set_mol_column(df=df, smiles_column=smiles_column, mol_column=mol_column)

        if self.include:
            df[self.mark_column] = df[self.names_column].apply(lambda x: len(x) > 0)
        elif not self.include:
            df[self.mark_column] = df[self.names_column].apply(lambda x: len(x) == 0)

        return self.mark_or_remove(df, mode, self.mark_column)
    
    def match_smarts_by_catalog(self, mol, smarts_list, names_list):
        """
        Match a SMARTS pattern against a molecule using a catalog.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            The molecule to match the SMARTS pattern against.
        smarts_list : list
            A list of SMARTS patterns to match.
        names_list : list
            A list of names corresponding to the SMARTS patterns.

        Returns
        -------
        dict
            A dictionary with names as keys and matched atom indices as values.
        """
        catalog = catalog_from_smarts(smarts_list, names_list)
        matches = catalog.GetMatches(mol)
        return [m.GetDescription() for m in matches]

class ProximityFilter(BaseFilter):
    """
    Filter class to filter a DataFrame based on the proximity of two sites of
    interest based on SMARTS.
    """
    smarts_list_a:list
    smarts_list_b:list

    names_list_a:list
    names_list_b:list
    
    smarts_column_a:str
    smarts_column_b:str

    mark_column: ClassVar[str] = "passed_proximity_filter"
    inter_col: ClassVar[str] = "inter_distances"

    def filter(self,
                df: pd.DataFrame,
                inter_col:str,
                min_dist:float=None,
                max_dist:float=None,
                smiles_column:str="OPENADMET_CANONICAL_SMILES",
                mol_column:str = "mol",
                mode:str="mark",
                calculate:bool=True) -> pd.DataFrame:
        """
        Filter out compounds where chromophore is greater than
        min_dist bonds away from all protonatable sites.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.
        inter_col : str
            The column name containing the distances between the two sites of interest.
        min_dist : float
            The minimum distance threshold for the filter.
        max_dist : float
            The maximum distance threshold for the filter.
        mode : str
            Either "mark" or "remove". If "mark", the filter will mark the rows that meet the criteria
            either True or False. If "remove", the filter will remove the rows that meet the criteria.

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
        """

        if calculate:
            df = self.calculate(df, inter_col=inter_col, smiles_column=smiles_column, mol_column=mol_column)

        df = self.set_mol_column(df=df, smiles_column=smiles_column, mol_column=mol_column)

        df = self.min_max_filter(df,
                            property=inter_col,
                            min_threshold=min_dist,
                            max_threshold=max_dist,
                            mark_column=self.mark_column)

        return self.mark_or_remove(df, mode, self.mark_column)

    def calculate(self,
                  df: pd.DataFrame,
                  inter_col:str,
                  smiles_column:str="OPENADMET_CANONICAL_SMILES",
                  mol_column:str = "mol") -> pd.DataFrame:

        df = self.set_mol_column(df=df, smiles_column=smiles_column, mol_column=mol_column)

        if not self.smarts_column_a in df.columns:
            df[self.smarts_column_a] = df[mol_column].apply(
                lambda x: self.match_smarts(x, self.smarts_list_a, self.names_list_a)
            )

        if not self.smarts_column_b in df.columns:
            df[self.smarts_column_b] = df[mol_column].apply(
                lambda x: self.match_smarts(x, self.smarts_list_b, self.names_list_b)
            )

        df[inter_col] = df.apply(lambda x: self.get_min_dist(x), axis=1)
            
        return df

    def match_smarts(self, mol, smarts_list, names_list):
        """
        Match a SMARTS pattern against a molecule.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            The molecule to match the SMARTS pattern against.
        smarts_list : list
            A list of SMARTS patterns to match.
        names_list : list
            A list of names corresponding to the SMARTS patterns.

        Returns
        -------
        list
            A list of SMARTS names from names_list that match the SMARTS pattern.
        """

        matches = {}
        for smarts, name in zip(smarts_list, names_list):
            s_mol = Chem.MolFromSmarts(smarts)
            match_atoms, _ = dm.substructure_matching_bonds(mol, s_mol)
            if match_atoms:
                matches[name] = match_atoms

        return matches

    def get_min_dist(self, df_row) -> list:
        mol = df_row["mol"]
        sites_a = df_row[self.smarts_column_a]
        sites_b = df_row[self.smarts_column_b]
        if not sites_a or not sites_b:
            return pd.NA
        dist_matrix = Chem.GetDistanceMatrix(mol)
        min_dist = np.inf

        # Flatten all atom indices from all matches in sites_a and sites_b
        atoms_a = [atom for match in sites_a.values() for match_tuple in match for atom in match_tuple]
        atoms_b = [atom for match in sites_b.values() for match_tuple in match for atom in match_tuple]

        for a in atoms_a:
            for b in atoms_b:
                d = dist_matrix[a, b]
                if d < min_dist:
                    min_dist = d
        return min_dist if min_dist != np.inf else pd.NA

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
    min_pka: float = Field(default=3, description="The minimum pKa value for the range check.")
    max_pka: float = Field(default=11, description="The maximum pKa value for the range check.")
    min_unit_sep: float = Field(default=1, description="The minimum unit separation between pKa values.")

    def filter(self, df: pd.DataFrame, pka_column: str = "pka", mode="mark", calculate=True) -> pd.DataFrame:
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
            df["pka_in_range"] = df[pka_column].apply(lambda x: self.pkas_valid_range(x))

        if self.min_unit_sep:
            # filter for pka values that are at least min_unit_sep apart
            df["pka_unit_sep"] = df[pka_column].apply(lambda x : self.pka_separation(x, self.min_unit_sep))

        return self.mark_or_remove(df, mode, ["pka_in_range", "pka_unit_sep"])

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

    def pka_separation(self, pkas: list, min_unit_sep: float) -> bool:
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
                if abs(pkas[i] - pkas[j]) < min_unit_sep:
                    return False
        return True

class DatamolFilter(BaseFilter):

    name: str = Field(description="Descriptor name to filter on.")
    min_value: float = Field(description="Minimum descriptor value for the filter.")
    max_value: float = Field(description="Maximum descriptor value for the filter.")
    name_options: list = ['mw','fsp3','n_hba','n_hbd','n_rings','n_hetero_atoms','n_heavy_atoms',
                          'n_rotatable_bonds','n_aliphatic_rings','n_aromatic_rings','n_saturated_rings',
                          'n_radical_electrons','tpsa','qed','clogp','sas']

    def __post_init__(self):
        if self.name not in self.name_options:
            raise ValueError(f"Descriptor name must be one of {self.name_options}.")

    def filter(self, 
               df: pd.DataFrame, 
               col_name:str, 
               mode="mark", 
               smiles_column:str="OPENADMET_CANONCIAL_SMILES", 
               mol_column="mol", 
               calculate=True) -> pd.DataFrame:
        
        if calculate:
            df = self.calculate(df, col_name, smiles_column, mol_column)
            
        df = self.set_mol_column(df=df, smiles_column=smiles_column, mol_column=mol_column)

        if col_name not in df.columns:
            raise ValueError(f"The DataFrame must contain a '{col_name}' column.")
        
        mark_column = f"passed_{col_name}_filter"
        df = self.min_max_filter(
            df=df,
            property=col_name,
            min_threshold=self.min_value,
            max_threshold=self.max_value,
            mark_column=mark_column
        )

        return self.mark_or_remove(df, mode, mark_column)

    def calculate(self, 
                  df: pd.DataFrame, 
                  col_name:str, 
                  smiles_column:str = "OPENADMET_CANONICAL_SMILES", 
                  mol_column:str = "mol") -> pd.DataFrame:
        """
        Calculate the descriptor values for the DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to calculate descriptor values for.
        smiles_column : str
            The column name containing SMILES strings (default is 'OPENADMET_CANONICAL_SMILES').

        Returns
        -------
        pandas.DataFrame
            The DataFrame with calculated descriptor values.
        """

        df = self.set_mol_column(df=df, smiles_column=smiles_column, mol_column=mol_column)

        df[col_name] = df[mol_column].apply(lambda x: eval(f"dm.descriptors.{self.name}")(x))

        return df

