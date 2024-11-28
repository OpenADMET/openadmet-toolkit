import chembl_downloader
from typing import Tuple, Optional, Union
from pathlib import Path
import duckdb
import pandas as pd
from pydantic import BaseModel, Field, field_validator



class ChEMBLDatabaseConnector(BaseModel):
    """
    Class to handle the ChEMBL SQLite database

    Provides read-only access to the ChEMBL SQLite database, either by downloading the database
    or by providing the path to an existing database.

    You can also connect to a remote ChEMBL SQLite database by providing the URL to the database file.
    """
    version: int = Field(..., description="Version of the ChEMBL SQLite database.")
    sqlite_path: Path = Field(..., description="Path to the ChEMBL SQLite database file.")
    _connection: Optional[duckdb.DuckDBPyConnection] = None



    @staticmethod
    def check_chembl_version(version: int) -> None:
        """Check if the ChEMBL version is valid."""
        if str(version) not in chembl_downloader.versions():
            raise ValueError(f"Invalid ChEMBL version: {version}. Available versions: {chembl_downloader.versions()}")

    @classmethod
    def create_chembl_database(cls, version: int) -> "ChEMBLDatabaseConnector":
        """
        Create a new ChEMBL database by downloading the SQLite database.

        Parameters
        ----------

        version: int
            Version of the ChEMBL database to download.

        Returns
        -------
        ChEMBLDatabaseConnector
            Instance of the ChEMBLDatabaseConnector class.
        """
        cls.check_chembl_version(version)
        path = chembl_downloader.download_extract_sqlite(version=str(version))
        return cls(version=version, sqlite_path=path)
    
    def _prep_duckdb(self) -> None:
        """
        Prepare the DuckDB connection by installing and loading the required extensions.
        Then let connection lapse, with 
        """
        con = duckdb.connect()
        con.install_extension("httpfs")
        con.load_extension("httpfs")
        con.install_extension("sqlite")
        con.load_extension("sqlite")


    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get the DuckDB connection to the ChEMBL database."""
        if self._connection is None:
            self._connection = self._connect()
        return self._connection

    
    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Connect to the ChEMBL SQLite database."""
        self._prep_duckdb()
        return duckdb.connect(str(self.sqlite_path))
    

    def query(self, sql: str, return_as="duckdb") -> Union[duckdb.DuckDBPyConnection, pd.DataFrame]:
        """Execute a SQL query on the ChEMBL database."""
        data =  self.connection.query(sql)
        if return_as == "duckdb":
            return data
        elif return_as == "df":
            return data.to_df()
        else:
            raise ValueError(f"Invalid return_as value: {return_as}. Use 'duckdb' or 'df'.")


    def sql(self, sql: str) -> None:
        """Execute a SQL command on the ChEMBL database."""
        self.connection.sql(sql)

    


class ChEMBLTargetCurator(BaseModel):
    chembl_target_id: str = Field(..., description="ChEMBL target ID.")
    standard_type: str = Field(..., description="Standard type of the ChEMBL data.")
    min_assay_size: int = Field(20, description="Minimum assay size to be considered.")
    max_assay_size: int = Field(1000, description="Maximum assay size to be considered.")
    only_docs: bool = Field(True, description="Consider only assays with associated documents.")
    remove_mutants: bool = Field(True, description="Remove assays with mutant targets as best as possible.")
    only_high_confidence: bool = Field(True, description="Consider only high confidence assays >= 9")


    chembl_version: int = Field(34, description="Version of the ChEMBL database.")
    _chembl_connector: Optional[ChEMBLDatabaseConnector] = None


    def __init__(self, **data):
        super().__init__(**data)
        self._chembl_connector = ChEMBLDatabaseConnector.create_chembl_database(version=self.chembl_version)

    


    @field_validator("standard_type")
    def check_in_allowed_standard_types(cls, value):
        allowed_standard_types = ["IC50", "Ki", "Kd", "EC50"]
        if value not in allowed_standard_types:
            raise ValueError(f"Invalid standard type: {value}. Allowed values: {allowed_standard_types}")
        return value

    

    def get_assays_for_target(self) -> pd.DataFrame:
        query = f"""
            drop table if exists temp_assay_data;
            create temporary table temp_assay_data as
            select assay_id,assays.chembl_id assay_chembl_id,description,tid,targets.chembl_id target_chembl_id,\
                    count(distinct(molregno)) molcount,pref_name,assays.doc_id doc_id,docs.year doc_date,variant_id, assays.confidence_score \
                    from activities  \
                    join assays using(assay_id)  \
                    join docs on (assays.doc_id = docs.doc_id)  \
                    join target_dictionary as targets using (tid) \

                    where pchembl_value is not null   \
                    and standard_type='{self.standard_type}' \
                    and standard_units = 'nM'  \
                    and data_validity_comment is null  \
                    and standard_relation = '=' \
                    and target_type = 'SINGLE PROTEIN' \
                    and target_chembl_id = '{self.chembl_target_id}' \
                    group by (assay_id,assays.chembl_id,description,tid,targets.chembl_id,pref_name,\
                            assays.doc_id,docs.year,variant_id, assays.confidence_score) \
                    order by molcount desc; 
            """
        
        self._chembl_connector.sql(query)
        
        if self.only_docs:
            query = """ 
            delete FROM temp_assay_data where doc_date is null;
            """
            self._chembl_connector.sql(query)

        if self.remove_mutants:
            query = """
                    delete from temp_assay_data where variant_id is not null or lower(description) like '%mutant%' \
                    or lower(description) like '%mutation%' or lower(description) like '%variant%';
            """

            self._chembl_connector.sql(query)
        
        if self.only_high_confidence:
            query = """
            alter table temp_assay_data rename to temp_assay_data1;
            drop table if exists temp_assay_data;
            create temporary table temp_assay_data as
            select ta1.* \
            from temp_assay_data1 ta1 join assays using(assay_id) \
            where assays.confidence_score = 9;
            drop table temp_assay_data1;
            """
            self._chembl_connector.sql(query)

        return self._chembl_connector.query("select * from temp_assay_data", return_as="df")

