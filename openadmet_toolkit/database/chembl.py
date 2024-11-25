import chembl_downloader
from typing import Tuple
from pathlib import Path
import duckdb
from pydantic import BaseModel, Field

def download_extract_chembl_sqlite(version=None) -> Tuple[str, Path]:
    """Download the ChEMBL SQLite database."""
    version, path =  chembl_downloader.download_extract_sqlite(version=version)
    return version, path



class ChemblSQLiteDatabase(BaseModel):
    """
    Class to handle the ChEMBL SQLite database

    Provides read-only access to the ChEMBL SQLite database, either by downloading the database
    or by providing the path to an existing database.

    You can also connect to a remote ChEMBL SQLite database by providing the URL to the database file.
    """
    version: int = Field(..., description="Version of the ChEMBL SQLite database.")
    sqlite_path: Path = Field(..., description="Path to the ChEMBL SQLite database file.")


    @classmethod
    def create_chembl_database(cls, version=None) -> ChemblSQLiteDatabase:
        version, path = download_extract_chembl_sqlite(version=version)
        return cls(version=version, sqlite_path=path)
    
    
    
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Connect to the ChEMBL SQLite database."""
        return duckdb.connect(str(self.sqlite_path))
    





