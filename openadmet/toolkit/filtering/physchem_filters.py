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

from pydantic import BaseModel, Field, field_validator
from openadmet.toolkit.filtering.filter_base import BaseFilter, FilterOutput
from typing import ClassVar

class SMARTSFilter(BaseFilter):
    """
    Filter class to filter a DataFrame based on SMARTS patterns.

    Parameters
    ----------
    smarts_list : list
        A list of SMARTS patterns to match against the molecules.
    names_list : list
        A list of names corresponding to the SMARTS patterns. i.e.
        The name for the SMARTS "cccccc" could be "benzene".
    include : bool, optional
        Whether to filter for or against the SMARTS patterns.
        If True, only molecules matching the SMARTS patterns will be included.
        If False, only molecules not matching the SMARTS patterns will be included.
        Default is True.
    """

    smarts_list: list
    names_list: list
    include: bool = True # whether to filter for or against smarts

    def filter(self,
               smiles:list) -> FilterOutput:
        """
        Filter a DataFrame based on SMARTS patterns.

        Parameters
        ----------
        smiles : list-like
            A list of SMILES strings to be filtered.

        Returns
        -------
        FilterOutput
            A FilterOutput object containing the filtered values and a
            boolean list indicating whether each row meets the filter criteria.
        """

        mols = self.get_mols(smiles=smiles)

        matches = self.calculate(smiles, mols)

        if self.include:
            passes = matches.apply(lambda x: len(x) > 0)
        elif not self.include:
            passes = matches.apply(lambda x: len(x) == 0)

        return FilterOutput(values=matches, passes=passes)

    def calculate(self,
                  smiles: list,
                  mols: list = None) -> list:
        """
        Calculate the matches of the SMARTS patterns against the molecules.

        Parameters
        ----------
        smiles : list-like
            A list of SMILES strings to be filtered.
        mols : list, optional
            A list of RDKit Mol objects corresponding to the SMILES strings.
            If not provided, it will be generated from the SMILES strings.

        Returns
        -------
        list
            A list containing the matches for each molecule.
        """

        if mols is None:
            mols = self.get_mols(smiles=smiles)

        matches = mols.apply(
            lambda x: self._match_smarts_by_catalog(x, self.smarts_list, self.names_list)
        )

        return matches

    def _match_smarts_by_catalog(self, mol, smarts_list, names_list):
        """
        Match a molecule against a list of SMARTS patterns using a catalog.
        This method uses the `catalog_from_smarts` function to create a catalog
        from the provided SMARTS patterns and names, and then retrieves matches.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            The RDKit molecule to match against the SMARTS patterns.
        smarts_list : list
            A list of SMARTS patterns to match against the molecule.
        names_list : list
            A list of names corresponding to the SMARTS patterns.

        Returns
        -------
        list
            A list of descriptions of the matches found in the molecule.
        """

        catalog = catalog_from_smarts(smarts_list, names_list)
        matches = catalog.GetMatches(mol)
        return [m.GetDescription() for m in matches]

class ProximityFilter(BaseFilter):
    """
    Filter class to filter molecules based on the proximity of two sets of SMARTS patterns.

    This filter checks the minimum distance between matches of two sets of SMARTS patterns
    within a specified distance range.

    Parameters
    ----------
    smarts_list_a : list
        A list of SMARTS patterns for the first set.
    smarts_list_b : list
        A list of SMARTS patterns for the second set.
    names_list_a : list
        A list of names corresponding to the SMARTS patterns in the first set.
        i.e. The name for the SMARTS "cccccc" could be "benzene".
    names_list_b : list
        A list of names corresponding to the SMARTS patterns in the second set.
        i.e. The name for the SMARTS "cccccc" could be "benzene".
    min_dist : int, optional
        The minimum distance, in number of atom bonds,
        between matches of the two sets of SMARTS patterns.
        If not specified, no minimum distance filter is applied.
    max_dist : int, optional
        The maximum distance, in number of atom bonds,
        between matches of the two sets of SMARTS patterns.
        If not specified, no maximum distance filter is applied.
    """

    smarts_list_a:list
    smarts_list_b:list

    names_list_a:list
    names_list_b:list

    min_dist:int=None
    max_dist:int=None

    def filter(self,
                smiles:list) -> FilterOutput:
        """
        Filter a DataFrame based on the proximity of two sets of SMARTS patterns.

        This method calculates the minimum distance between matches of the two sets
        of SMARTS patterns and checks if they fall within the specified distance range.

        Parameters
        ----------
        smiles : list-like
            A list of SMILES strings to be filtered.

        Returns
        -------
        FilterOutput
            A FilterOutput object containing the calculated distances and a boolean list
            indicating whether each row meets the filter criteria based on the specified distance range.
        """

        mols = self.get_mols(smiles=smiles)

        distances = self.calculate(smiles=smiles, mols=mols)

        passes = self.min_max_filter(property=distances,
                            min_threshold=self.min_dist,
                            max_threshold=self.max_dist)

        return FilterOutput(values=distances, passes=passes)

    def calculate(self,
                  smiles:list,
                  mols:list = None) -> list:
        """
        Calculate the minimum distance between matches of two sets of SMARTS patterns.
        This method applies the SMARTS matching to each molecule and computes the minimum distance
        between matches of the two sets of SMARTS patterns.

        Parameters
        ----------
        smiles : list-like
            A list of SMILES strings to be filtered.
        mols : list, optional
            A list of RDKit Mol objects corresponding to the SMILES strings.
            If not provided, it will be generated from the SMILES strings.

        Returns
        -------
        list
            A list containing the minimum distances between matches of the two sets of SMARTS patterns
            for each molecule. If no matches are found, it returns pd.NA.
        """

        if mols is None:
            mols = self.set_mols(smiles=smiles)

        matches_a = mols.apply(
                lambda x: self._match_smarts(x, self.smarts_list_a, self.names_list_a)
            )

        matches_b = mols.apply(
                lambda x: self._match_smarts(x, self.smarts_list_b, self.names_list_b)
            )

        df = pd.DataFrame({
            "mol": mols,
            "col_a": matches_a,
            "col_b": matches_b
        })

        inter_dists = df.progress_apply(self._get_min_dist, axis=1)

        return inter_dists

    def _match_smarts(self, mol, smarts_list, names_list) -> dict:
        """
        Match a molecule against a list of SMARTS patterns and return the matches.
        This method uses RDKit to create a Mol object from each SMARTS pattern
        and checks for substructure matches in the provided molecule.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            The RDKit molecule to match against the SMARTS patterns.
        smarts_list : list
            A list of SMARTS patterns to match against the molecule.
        names_list : list
            A list of names corresponding to the SMARTS patterns.

        Returns
        -------
        dict
            A dictionary where keys are names of the SMARTS patterns and values are lists of tuples
            containing the atom indices of the matches in the molecule.
        """

        matches = {}
        for smarts, name in zip(smarts_list, names_list):
            s_mol = Chem.MolFromSmarts(smarts)
            match_atoms, _ = dm.substructure_matching_bonds(mol, s_mol)
            if match_atoms:
                matches[name] = match_atoms

        return matches

    def _get_min_dist(self, df_row) -> list:
        """
        Calculate the minimum distance between matches of two sets of SMARTS patterns
        for a given row in the DataFrame.

        This method retrieves the RDKit Mol object and the matches for both sets of SMARTS patterns,
        computes the distance matrix, and finds the minimum distance between all pairs of matches.

        Parameters
        ----------
        df_row : pd.Series
            A row from the DataFrame containing the RDKit Mol object and the matches for both sets
            of SMARTS patterns.

        Returns
        -------
        float or pd.NA
            The minimum distance between matches of the two sets of SMARTS patterns.
            If no matches are found in either set, it returns pd.NA.
        """

        mol = df_row["mol"]
        sites_a = df_row["col_a"]
        sites_b = df_row["col_b"]
        if not sites_a or not sites_b:
            return pd.NA
        dist_matrix = Chem.GetDistanceMatrix(mol)
        self.min_dist = np.inf

        # Flatten all atom indices from all matches in sites_a and sites_b
        atoms_a = [atom for match in sites_a.values() for match_tuple in match for atom in match_tuple]
        atoms_b = [atom for match in sites_b.values() for match_tuple in match for atom in match_tuple]

        for a in atoms_a:
            for b in atoms_b:
                d = dist_matrix[a, b]
                if d < self.min_dist:
                    self.min_dist = d

        return self.min_dist if self.min_dist != np.inf else pd.NA

class pKaFilter(BaseFilter):
    """
    Filter class to filter molecules based on pKa values.

    This filter checks if at least one pKa value is within a specified range
    and if all pKa values are separated by a minimum distance.

    Parameters
    ----------
    min_pka : float, optional
        The minimum pKa value for the range check. Default is 3.
    max_pka : float, optional
        The maximum pKa value for the range check. Default is 11.
    min_unit_sep : float, optional
        The minimum unit separation between pKa values. Default is 1.
    """

    min_pka: float = Field(default=3, description="The minimum pKa value for the range check.")
    max_pka: float = Field(default=11, description="The maximum pKa value for the range check.")
    min_unit_sep: float = Field(default=1, description="The minimum unit separation between pKa values.")

    def filter(self,
               pkas: list) -> FilterOutput:
        """
        Filter a DataFrame based on pKa values.

        This method checks if at least one pKa value is within the specified range
        and if all pKa values are separated by a minimum distance.

        Parameters
        ----------
        pkas : list
            A list of pKa values for each molecule to be filtered.

        Returns
        -------
        FilterOutput
            A FilterOutput object containing the pKa values and a boolean list
            indicating whether each row meets the filter criteria based on the specified pKa range and separation.
        """

        if self.min_pka and self.max_pka:
            # filter for at least one pka between min_pka and max_pka
            pkas_min_max = pkas.apply(self._pkas_valid_range)

        if self.min_unit_sep:
            # filter for pka values that are at least min_unit_sep apart
            pkas_unit_sep = pkas_min_max.apply(lambda x : self._pka_separation(x, self.min_unit_sep))

        return FilterOutput(values=pkas,
            passes=pkas_min_max & pkas_unit_sep)

    def _pkas_valid_range(self, pkas: list) -> bool:
        """
        Check if at least one pKa value is within the specified range.

        This method iterates through the list of pKa values and checks if any of them
        fall within the range defined by `min_pka` and `max_pka`.

        Parameters
        ----------
        pkas : list
            A list of pKa values to be checked.

        Returns
        -------
        bool
            True if at least one pKa value is within the range, False otherwise.
        """

        valid_range = False
        for pka in pkas:
            if self.min_pka <= pka <= self.max_pka:
                valid_range = True
                break
        return valid_range

    def _pka_separation(self, pkas: list, min_unit_sep: float) -> bool:
        """
        Check if all pKa values are separated by a minimum distance.

        This method iterates through the list of pKa values and checks if the absolute difference
        between any two pKa values is greater than or equal to `min_unit_sep`.

        Parameters
        ----------
        pkas : list
            A list of pKa values to be checked.
        min_unit_sep : float
            The minimum distance that should separate any two pKa values.

        Returns
        -------
        bool
            True if all pKa values are separated by at least `min_unit_sep`, False otherwise.
        """

        for i in range(len(pkas)):
            for j in range(i + 1, len(pkas)):
                if abs(pkas[i] - pkas[j]) < min_unit_sep:
                    return False
        return True

class DatamolFilter(BaseFilter):
    """
    Filter class to filter molecules based on physicochemical descriptors using Datamol.

    This filter calculates a specified descriptor for each molecule and checks if
    the descriptor value falls within a specified range.

    Parameters
    ----------
    name : str
        The name of the descriptor to filter on. Must be one of the predefined options.
    min_value : float
        The minimum descriptor value for the filter.
    max_value : float
        The maximum descriptor value for the filter.
    """

    name: str = Field(description="Descriptor name to filter on.")
    min_value: float = Field(description="Minimum descriptor value for the filter.")
    max_value: float = Field(description="Maximum descriptor value for the filter.")

    @field_validator('name')
    @classmethod
    def validate_name(cls, value):
        """
        Validate that the provided name is one of the allowed descriptor names.

        """
        name_options = ['mw','fsp3','n_hba','n_hbd','n_rings','n_hetero_atoms','n_heavy_atoms',
                          'n_rotatable_bonds','n_aliphatic_rings','n_aromatic_rings','n_saturated_rings',
                          'n_radical_electrons','tpsa','qed','clogp','sas']
        if value not in name_options:
            raise ValueError(f"Descriptor name must be one of {name_options}.")
        return value

    def filter(self,
               smiles:list) -> FilterOutput:
        """
        Filter a DataFrame based on a physicochemical descriptor calculated using Datamol.

        This method calculates the specified descriptor for each molecule and checks if
        the descriptor value falls within the specified range.

        Parameters
        ----------
        smiles : list-like
            A list of SMILES strings to be filtered.

        Returns
        -------
        FilterOutput
            A FilterOutput object containing the calculated descriptor values and a boolean list
            indicating whether each row meets the filter criteria based on the specified descriptor range.
        """

        mols = self.get_mols(smiles=smiles)

        property = self.calculate(smiles, mols)

        passes = self.min_max_filter(
            property=property,
            min_threshold=self.min_value,
            max_threshold=self.max_value,
        )

        return FilterOutput(values=property, passes=passes)

    def calculate(self,
                  smiles:list,
                  mols:list = None) -> list:
        """
        Calculate the specified physicochemical descriptor for each molecule.

        This method uses Datamol to compute the descriptor value for each molecule
        based on the provided SMILES strings or RDKit Mol objects.

        Parameters
        ----------
        smiles : list-like
            A list of SMILES strings to be filtered.
        mols : list, optional
            A list of RDKit Mol objects corresponding to the SMILES strings.
            If not provided, it will be generated from the SMILES strings.

        Returns
        -------
        list
            A list containing the calculated descriptor values for each molecule.
            If no molecules are provided, it returns an empty list.
        """

        if mols is None:
            mols = self.get_mols(smiles=smiles)

        property = mols.apply(lambda x: getattr(dm.descriptors, self.name)(x))

        return property
