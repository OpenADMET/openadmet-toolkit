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

from pydantic import BaseModel, Field
from openadmet.toolkit.filtering.filter_base import BaseFilter

def min_max_filter(df: pd.DataFrame,
                   property: str, 
                   min_threshold: float, 
                   max_threshold: float, 
                   mark_column: str) -> bool:
    """
    Filter a DataFrame based on a property value range.

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
    mark_column : str
        The name of the column to store the boolean marks (True/False).

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only rows where the property values are within the specified range.
    """
    if min_threshold is not None and max_threshold is None:
        df[mark_column] = df[property] >= min_threshold
    elif max_threshold is not None and min_threshold is None:
        df[mark_column] = df[property] <= max_threshold
    elif min_threshold is not None and max_threshold is not None:
        df[mark_column] = (df[property] >= min_threshold) & (df[property] <= max_threshold)
    else:
        raise ValueError("Either min_threshold or max_threshold must be provided.")
    
    return df

class SMARTSFilter(BaseFilter):
    """
    Filter class to filter a DataFrame based on SMARTS patterns.

    """
    smarts: str = Field(description="SMARTS pattern to filter the DataFrame.")
    
    def filter(self, df: pd.DataFrame, mode="mark") -> pd.DataFrame:
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
        if "smiles" not in df.columns:
            raise ValueError("The DataFrame must contain a 'smiles' column.")

        

        
        return df


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
    
    def filter(self, df: pd.DataFrame, mode="mark") -> pd.DataFrame:
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
        if "pka" not in df.columns:
            raise ValueError("The DataFrame must contain a 'pka' column.")
        
        # filter for at least one pka between min_pka and max_pka

        
        # filter the pkas to be at least min_unit_sep apart


        return self.mark_or_remove(df, mode, "pka_filtered")
    
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
            The input DataFrame to be filtered.
        mode : str
            Either "mark" or "remove". If "mark", the filter will mark the rows that meet the criteria
            either True or False. If "remove", the filter will remove the rows that meet the criteria.
        
        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
        """
        # check if the logP column exists
        if "logP" not in df.columns:
            raise ValueError("The DataFrame must contain a 'logP' column.")
        
        # filter for logP values between min_logP and max_logP
        df = min_max_filter(df, "logP", self.min_logP, self.max_logP, "logP_filtered")

        return self.mark_or_remove(df, mode, "logP_filtered")
