import pandas as pd
from pydantic import BaseModel, Field

class BaseFilter(BaseModel):
    """
    Base class for filtering chemical data based on various properties.
    """
    @abstractmethod
    def filter(self, df: pd.DataFrame, mode="mark") -> pd.DataFrame:
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
