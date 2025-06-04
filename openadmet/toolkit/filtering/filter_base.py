import pandas as pd
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from rdkit import Chem

def min_max_filter(df: pd.DataFrame,
                   property: str,
                   min_threshold: float,
                   max_threshold: float,
                   mark_column: str,
                   any_or_all: str = None) -> bool:
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
    any_or_all : str, optional
        If "any", the mark will be True if any of the conditions are met.
        If "all", the mark will be True only if all conditions are met.
        Default is pd.NA, which means no specific condition is applied.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing only rows where the property values are within the specified range.
    """
    if min_threshold is not None and max_threshold is None:
        condition = df[property] >= min_threshold
    elif max_threshold is not None and min_threshold is None:
        condition = df[property] <= max_threshold
    elif min_threshold is not None and max_threshold is not None:
        condition = (df[property] >= min_threshold) & (df[property] <= max_threshold)
    else:
        raise ValueError("Either min_threshold or max_threshold must be provided.")
    
    if not any_or_all:
        df[mark_column] = condition
    else:
        if any_or_all == "any":
            df[mark_column] = condition.any(axis=1)
        elif any_or_all == 'all':
            df[mark_column] = condition.all(axis=1)

    return df

def mark_or_remove(df: pd.DataFrame, mode:str, mark_columns = None) -> pd.DataFrame:
    """
    Remove rows from a DataFrame that are marked as True in a specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame to be filtered.
    mark_columns : str
        The name of the column containing the boolean marks.

    Returns
    -------
    pandas.DataFrame
        The filtered DataFrame with marked rows removed.
    """
    if isinstance(mark_columns, str):
        mark_columns = [mark_columns]
    if mode == "remove":
        for mark_col in mark_columns:
            if mark_col not in df.columns:
                raise ValueError(f"Column {mark_col} not found in DataFrame.")
            df = df[df[mark_col] == False].drop(columns=[mark_col])
    elif mode != "mark":
        raise ValueError("mode must be either 'mark' or 'remove'")
    return df

class BaseFilter(BaseModel):
    """
    Base class for filtering chemical data based on various properties.
    """
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def filter(self, df: pd.DataFrame, mode:str="mark") -> pd.DataFrame:
        """
        Abstract method to filter a DataFrame based on specific criteria.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.
        mode : str, optional
            Either "mark" or "remove". If "mark", the filter will mark the rows that meet the criteria
            either True or False.
            If "remove", the filter will remove the rows that meet the criteria. Default is "mark".

        Returns
        -------
        pandas.DataFrame
            The filtered DataFrame.
        """
        pass

    @staticmethod
    def set_mol_column(df: pd.DataFrame, smiles_column: str = "OPENADMET_CANONICAL_SMILES", mol_column: str = "mol") -> pd.DataFrame:
        """
        Set the mol column in the DataFrame if it does not exist.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame to be filtered.
        mol_column : str
            The column name containing the RDKit Mol objects (default is 'mol').

        Returns
        -------
        pandas.DataFrame
            The DataFrame with the mol column set.
        """

        if smiles_column not in df.columns:
            raise ValueError(f"The DataFrame must contain a {smiles_column} column.")

        df[mol_column] = df[smiles_column].apply(
                lambda x: Chem.MolFromSmiles(x)
            )
        return df
