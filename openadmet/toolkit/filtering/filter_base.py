import pandas as pd
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod

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
