import abc
from pathlib import Path
from typing import Optional, Tuple, Union

import chembl_downloader
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
    sqlite_path: Path = Field(
        ..., description="Path to the ChEMBL SQLite database file."
    )
    _connection: Optional[duckdb.DuckDBPyConnection] = None

    @staticmethod
    def check_chembl_version(version: int) -> None:
        """Check if the ChEMBL version is valid."""
        if str(version) not in chembl_downloader.versions():
            raise ValueError(
                f"Invalid ChEMBL version: {version}. Available versions: {chembl_downloader.versions()}"
            )

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
        # cls.check_chembl_version(version)
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

    def query(
        self, sql: str, return_as="duckdb"
    ) -> Union[duckdb.DuckDBPyConnection, pd.DataFrame]:
        """Execute a SQL query on the ChEMBL database."""
        data = self.connection.query(sql)
        if return_as == "duckdb":
            return data
        elif return_as == "df":
            return data.to_df()
        else:
            raise ValueError(
                f"Invalid return_as value: {return_as}. Use 'duckdb' or 'df'."
            )

    def sql(self, sql: str) -> None:
        """Execute a SQL command on the ChEMBL database."""
        self.connection.sql(sql)


class ChEMBLTargetCuratorBase(BaseModel):
    chembl_target_id: str = Field(..., description="ChEMBL target ID.")
    chembl_version: int = Field(34, description="Version of the ChEMBL database.")
    _chembl_connector: Optional[ChEMBLDatabaseConnector] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._chembl_connector = ChEMBLDatabaseConnector.create_chembl_database(
            version=self.chembl_version
        )

    @abc.abstractmethod
    def get_activity_data(
        self, return_as="df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """Get the activity data for a given target using its ChEMBL ID."""
        pass

    @abc.abstractmethod
    def aggregate_activity_data_by_compound(self) -> pd.DataFrame:
        """Get the activity data for a given target using its ChEMBL ID, grouped by compound."""
        pass


class HighQualityChEMBLTargetCurator(ChEMBLTargetCuratorBase):
    """
    Implement ChEMBL curation for a given protein target.

    Curation rules are taken from https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049
    with accompanying code here: https://github.com/rinikerlab/overlapping_assays/tree/main

    Thank you @greglandrum!
    """

    standard_type: Optional[str] = Field(
        None,
        description="Select a single Standard type of the ChEMBL data (IC50, Ki, Kd, EC50).",
    )
    min_assay_size: int = Field(10, description="Minimum assay size to be considered.")
    max_assay_size: int = Field(
        10000, description="Maximum assay size to be considered."
    )
    only_docs: bool = Field(
        True, description="Consider only assays with associated documents."
    )
    remove_mutants: bool = Field(
        True, description="Remove assays with mutant targets as best as possible."
    )
    only_high_confidence: bool = Field(
        True, description="Consider only high confidence assays >= 9"
    )
    extra_filter: Optional[str] = Field(
        None,
        description="Extra filters to apply to the query, single word OR matching against assay description.",
    )
    landrum_curation: bool = Field(
        True, description="Apply the maximum curation rules from the Landrum paper."
    )
    landrum_activity_curation: bool = Field(
        False, description="curation of activity data according to Landrum rules."
    )
    landrum_no_duplicate_docs: bool = Field(
        False, description="Remove assays with duplicate documents."
    )
    landrum_overlap_min: int = Field(
        0, description="Minimum overlap between assays to be considered."
    )

    @field_validator("standard_type")
    def check_in_allowed_standard_types(cls, value):
        allowed_standard_types = ["IC50", "Ki", "Kd", "EC50"]
        if value not in allowed_standard_types:
            raise ValueError(
                f"Invalid standard type: {value}. Allowed values: {allowed_standard_types}"
            )
        return value

    def get_high_quality_assays_for_target(
        self, return_as="df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """
        Get the high quality assays for a given target using its ChEMBL ID.

        Parameters
        ----------
        return_as: str
            Return the data as a DataFrame or a DuckDB relation.
        """
        # first we get the assays for the target using a resonable level of curation.
        query = f"""
            drop table if exists temp_assay_data;
            create temporary table temp_assay_data as
            select assay_id,assays.chembl_id assay_chembl_id,description,tid,targets.chembl_id target_chembl_id,\
                    count(distinct(molregno)) molcount,pref_name,assays.doc_id doc_id,docs.year doc_date,variant_id, assays.confidence_score, standard_type \
                    from activities  \
                    join assays using(assay_id)  \
                    join docs on (assays.doc_id = docs.doc_id)  \
                    join target_dictionary as targets using (tid) \
                    where pchembl_value is not null   \
                    and standard_type is not null \
                    and standard_units = 'nM'  \
                    and data_validity_comment is null  \
                    and standard_relation = '=' \
                    and target_type = 'SINGLE PROTEIN' \
                    and target_chembl_id = '{self.chembl_target_id}' \
                    group by (assay_id,assays.chembl_id,description,tid,targets.chembl_id,pref_name,\
                            assays.doc_id,docs.year,variant_id, assays.confidence_score, standard_type) \
                    order by molcount desc;
            """

        self._chembl_connector.sql(query)

        # if we want a single standard type, we can filter it here
        if self.standard_type:
            query = f"""
            delete from temp_assay_data where standard_type != '{self.standard_type}';
            """
            self._chembl_connector.sql(query)

        # remove assays without documents
        if self.only_docs:
            query = """
            delete from temp_assay_data where doc_date is null;
            """
            self._chembl_connector.sql(query)

        # remove mutants
        if self.remove_mutants:
            query = """
                    delete from temp_assay_data where variant_id is not null or lower(description) like '%mutant%' \
                    or lower(description) like '%mutation%' or lower(description) like '%variant%';
            """

            self._chembl_connector.sql(query)

        # remove low confidence assays
        if self.only_high_confidence:
            query = """
            delete from temp_assay_data where confidence_score < 9;
            """
            self._chembl_connector.sql(query)

        # apply extra filter if provided
        if self.extra_filter:
            query = f"""
            delete from temp_assay_data where lower(description) not like '%{self.extra_filter}%';
            """
            self._chembl_connector.sql(query)

        # okay now find goldilocks zone assays between min and max assay size
        query = f"""
        delete from temp_assay_data where molcount >= {self.min_assay_size} and molcount <= {self.max_assay_size};
        """
        self._chembl_connector.sql(query)

        # get the final data
        query = """
        select * from temp_assay_data;
        """
        return self._chembl_connector.query(query, return_as=return_as)

    def get_activity_data(
        self, return_as="df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """
        Get the high quality activity data for a given target using its ChEMBL ID.
        """
        hq_assays = self.get_high_quality_assays_for_target(return_as="duckdb")
        # get all the activity data for the target
        qtext = f"""
        SELECT
        activities.assay_id                  AS assay_id,
        activities.doc_id                    AS doc_id,
        activities.standard_value            AS standard_value,
        molecule_hierarchy.parent_molregno   AS molregno,
        compound_structures.canonical_smiles AS canonical_smiles,
        target_dictionary.tid                AS tid,
        target_dictionary.chembl_id          AS target_chembl_id,
        pchembl_value                        AS pchembl_value
        FROM activities
        JOIN assays ON activities.assay_id = assays.assay_id
        JOIN target_dictionary ON assays.tid = target_dictionary.tid
        JOIN target_components ON target_dictionary.tid = target_components.tid
        JOIN component_class ON target_components.component_id = component_class.component_id
        JOIN molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        JOIN molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        JOIN compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        WHERE activities.standard_units = 'nM' AND
            pchembl_value IS NOT NULL AND
            activities.data_validity_comment IS NULL AND
            activities.standard_relation = '=' AND
            activities.potential_duplicate = 0 AND
            assays.confidence_score >= 8 AND
            target_dictionary.target_type = 'SINGLE PROTEIN' AND
            target_chembl_id = '{self.chembl_target_id}'
            """

        all_data = self._chembl_connector.query(qtext, return_as="duckdb")

        # find intersection of high quality assays and all data
        hq_data = hq_assays.join(all_data, "assay_id, tid, target_chembl_id, doc_id")

        if return_as == "df":
            return hq_data.to_df()
        else:
            return hq_data

    def aggregate_activity_data_by_compound(self) -> pd.DataFrame:
        """
        Get the high quality activity data for a given target using its ChEMBL ID, grouped by compound.
        """
        hq_data = self.get_activity_data(return_as="df")

        data = hq_data.groupby(["molregno", "canonical_smiles"]).agg(
            {
                "assay_id": "count",
                "standard_value": ["mean", "std"],
                "pchembl_value": ["mean", "std"],
            }
        )
        # unnenest column hierarchy
        data.columns = ["_".join(col) for col in data.columns]
        data = data.reset_index()
        return data


class PermissiveChEMBLTargetCurator(ChEMBLTargetCuratorBase):
    """
    Implement ChEMBL curation for a given protein target.

    This is a permissive curation, where we don't apply as many filters to the data
    we require a pChEMBL value and a standard value in nM (redundant anyway).


    """

    standard_type: Optional[str] = Field(
        None,
        description="Select a single Standard type of the ChEMBL data (IC50, Ki, Kd, EC50, Potency).",
    )
    extra_filter: Optional[str] = Field(
        None,
        description="Extra filters to apply to the query, single word OR matching against assay description.",
    )
    require_pchembl: bool = Field(True, description="Require a pChEMBL value.")

    @field_validator("standard_type")
    def check_in_allowed_standard_types(cls, value):
        allowed_standard_types = ["IC50", "Ki", "Kd", "EC50"]
        if value not in allowed_standard_types:
            raise ValueError(
                f"Invalid standard type: {value}. Allowed values: {allowed_standard_types}"
            )
        return value

    def get_activity_data(
        self, return_as="df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """
        Get all the activity data for a given target using its ChEMBL ID.
        """
        qtext = f"""

        SELECT
        activities.assay_id                  AS assay_id,
        activities.doc_id                    AS doc_id,
        activities.standard_value            AS standard_value,
        molecule_hierarchy.parent_molregno   AS molregno,
        compound_structures.canonical_smiles AS canonical_smiles,
        target_dictionary.tid                AS tid,
        target_dictionary.chembl_id          AS target_chembl_id,
        pchembl_value                        AS pchembl_value
        FROM activities
        JOIN assays ON activities.assay_id = assays.assay_id
        JOIN target_dictionary ON assays.tid = target_dictionary.tid
        JOIN target_components ON target_dictionary.tid = target_components.tid
        JOIN component_class ON target_components.component_id = component_class.component_id
        JOIN molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        JOIN molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        JOIN compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        WHERE activities.standard_units = 'nM' AND
            target_chembl_id = '{self.chembl_target_id}'
        """

        if self.require_pchembl:
            qtext += "AND pchembl_value IS NOT NULL"

        # if we specified a single standard type, we can filter it here as doesn't happen at the assay level
        if self.standard_type:
            qtext += f"AND standard_type = '{self.standard_type}'"

        all_data = self._chembl_connector.query(qtext, return_as="duckdb")

        if return_as == "df":
            return all_data.to_df()
        else:
            return all_data

    def aggregate_activity_data_by_compound(self) -> pd.DataFrame:
        """
        Get all the activity data for a given target using its ChEMBL ID, grouped by compound.
        """
        all_data = self.get_activity_data(return_as="df")
        data = all_data.groupby(["molregno", "canonical_smiles"]).agg(
            {
                "assay_id": "count",
                "standard_value": ["mean", "std"],
                "pchembl_value": ["mean", "std"],
            }
        )
        # unnenest column hierarchy
        data.columns = ["_".join(col) for col in data.columns]
        data = data.reset_index()
        return data

    def get_variant_ids_for_target(
        self, return_as: str = "df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """
        Get all the variant IDs for a given target using its ChEMBL ID
        Only target_type = 'SINGLE PROTEIN' curation is applied.

        Parameters
        ----------
        return_as: str
            Return the data as a DataFrame or a DuckDB relation.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the variant IDs for the target

        """
        query = f"""
        select distinct(variant_id), tid, description,  targets.chembl_id as target_chembl_id, assay_id, assays.chembl_id assay_chembl_id \
        from activities \
        join assays using(assay_id)  \
        join docs on (assays.doc_id = docs.doc_id)  \
        join target_dictionary as targets using (tid) \
        where target_type = 'SINGLE PROTEIN' \
        and target_chembl_id = '{self.chembl_target_id}' \
        and variant_id is not null
        group by (variant_id, tid, description, target_chembl_id, assay_id, assay_chembl_id) \
        """
        return self._chembl_connector.query(query, return_as=return_as)
