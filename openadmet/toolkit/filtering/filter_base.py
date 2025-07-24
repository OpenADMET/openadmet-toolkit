import pandas as pd
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from rdkit import Chem
from loguru import logger

class FilterOutput(BaseModel):
    """
    A class to hold the output of a filter operation.
    It contains the values that were filtered and a boolean list indicating
    whether each row meets the filter criteria.
    """

    values: list = Field(
        default=None,
        description="A list of values for the filter criteria."
    )
    passes: list = Field(
        default=None,
        description="A list of boolean values indicating whether each row meets the filter criteria."
    )

    def get_values(self) -> list:
        """
        Get the values for the filter criteria.

        Returns
        -------
        list
            A list of values for the filter criteria.
        """
        return self.values

    def get_passes(self) -> list:
        """
        Get the marks indicating whether each row meets the filter criteria.

        Returns
        -------
        list
            A list of boolean values indicating whether each row meets the filter criteria.
        """
        return self.passes

class BaseFilter(BaseModel):
    """
    Base class for filtering chemical data based on various properties.
    """
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def filter(self, smiles:list) -> FilterOutput:
        """
        Abstract method to filter a DataFrame based on specific criteria.

        Parameters
        ----------
        smiles : list
            A list of SMILES strings to be filtered.

        Returns
        -------
        FilterOutput
            An instance of FilterOutput containing the filtered results.
        """
        pass

    @staticmethod
    def get_mols(smiles: list) -> list:
        """
        Set the mol column in the DataFrame if it does not exist.

        Parameters
        ----------
        smiles : list
            A list of SMILES strings.

        Returns
        -------
        list
            A list of RDKit Mol objects corresponding to the SMILES strings.
        """
        mols = smiles.apply(
                lambda x: Chem.MolFromSmiles(x)
            )
        return mols

    @staticmethod
    def min_max_filter(property: pd.Series,
                    min_threshold: float,
                    max_threshold: float) -> pd.Series:
        """
        Filter a property based on minimum and maximum thresholds.

        Parameters
        ----------
        property : pd.Series
            The property to filter, typically a pandas Series.
        min_threshold : float
            The minimum threshold for the property.
        max_threshold : float
            The maximum threshold for the property.

        Returns
        -------
        pd.Series
            A boolean Series indicating whether each value in the property meets the filter criteria.
        """

        if min_threshold is not None and max_threshold is None:
            condition = property >= min_threshold
        elif max_threshold is not None and min_threshold is None:
            condition = property <= max_threshold
        elif min_threshold is not None and max_threshold is not None:
            condition = (property >= min_threshold) & (property <= max_threshold)
        else:
            raise ValueError("Either min_threshold or max_threshold must be provided.")

        return condition
