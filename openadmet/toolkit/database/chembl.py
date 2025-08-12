import abc
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import chembl_downloader
import datamol as dm
import duckdb
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator
from jinja2 import Template


from openadmet.toolkit.chemoinformatics.rdkit_funcs import canonical_smiles, smiles_to_inchikey


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
        # cls.check_chembl_version(version)  # sometimes flaky if chembl webservice is slow
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
        self, sql: str, return_as: str = "duckdb"
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


class ChEMBLCuratorBase(BaseModel):
    chembl_version: int = Field(34, description="Version of the ChEMBL database.")
    _chembl_connector: Optional[ChEMBLDatabaseConnector] = None

    def __init__(self, **data):
        super().__init__(**data)
        self._chembl_connector = ChEMBLDatabaseConnector.create_chembl_database(
            version=self.chembl_version
        )


    @abc.abstractmethod
    def get_templated_query(self) -> str:
        """
        Get the templated query for the data to pull from the ChEMBL database.
        This should be implemented by subclasses to return a valid SQL query string.
        """


    def get_activity_data(
        self, return_as: str = "df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """Get the activity data for a  query"""
        query = self.get_templated_query()

        all_data = self._chembl_connector.query(query, return_as="duckdb")

        if return_as == "df":
            return all_data.to_df()
        else:
            return all_data


    def aggregate_activity_data_by_compound(self, canonicalise=False) -> pd.DataFrame:
        """
        Aggregate the activity data by compound from a ChEMBL query.

        Parameters
        ----------
        canonicalise: bool
            Canonicalise the SMILES and calculate the InChIKey

        Returns
        -------
        pd.DataFrame
            DataFrame containing the aggregated activity data.
        """
        all_data = self.get_activity_data(return_as="df")
        if canonicalise:
            with dm.without_rdkit_log():
                all_data["OPENADMET_CANONICAL_SMILES"] = all_data[
                    "canonical_smiles"
                ].progress_apply(lambda x: canonical_smiles(x))
                all_data["OPENADMET_INCHIKEY"] = all_data[
                    "OPENADMET_CANONICAL_SMILES"
                ].progress_apply(lambda x: smiles_to_inchikey(x))
                data = all_data.groupby(["OPENADMET_CANONICAL_SMILES", "OPENADMET_INCHIKEY"]).agg(
                    {
                        "assay_id": "count",
                        "standard_value": ["mean", "median", "std"],
                        "pchembl_value": ["mean", "median", "std"],
                    }
                )
        else:
            data = all_data.groupby(["molregno", "canonical_smiles"]).agg(
                {
                    "assay_id": "count",
                    "standard_value": ["mean", "median", "std"],
                    "pchembl_value": ["mean", "median", "std"],
                    "compound_name": "first",  # or "unique" if you want all names
                }
            )
        # unnenest column hierarchy
        data.columns = ["_".join(col) for col in data.columns]
        data = data.reset_index()
        data.sort_values("assay_id_count", ascending=False, inplace=True)

        return data

    def get_activity_data_for_compounds(
        self, compounds: Iterable[str], canonicalise=False
    ) -> pd.DataFrame:
        """
        Get activities for a list of compounds from the ChEMBL database for the set curation.
        """
        # convert list of smiles to INCHIKEY
        if canonicalise:
            with dm.without_rdkit_log():
                inchikeys = [smiles_to_inchikey(canonical_smiles(x)) for x in compounds]
        else:
            inchikeys = [smiles_to_inchikey(x) for x in compounds]
        # get all the activity data for the target
        df = self.get_activity_data(return_as="df")
        subset = df[df["standard_inchi_key"].isin(inchikeys)]
        return subset





class ChEMBLTargetCuratorBase(ChEMBLCuratorBase):
    """
    Base class for ChEMBL target curation.

    This class provides a base implementation for curating ChEMBL data for a **specific protein target**.
    Subclasses should implement the `get_templated_query` method to provide the SQL query for the target.
    """

    chembl_target_id: str = Field(..., description="ChEMBL target ID.")

    standard_type: Optional[str] = Field(
        None,
        description="Select a single Standard type of the ChEMBL data (IC50, Ki, Kd, EC50, Potency), leave empty for all types.",
    )

    require_units: Optional[str] = Field(
        "nM", description="Require a specific unit for the standard value."
    )





    @field_validator("standard_type")
    def check_in_allowed_standard_types(cls, value):
        allowed_standard_types = ["IC50", "Ki", "Kd", "EC50", "AC50", "Potency"]
        if value not in allowed_standard_types:
            raise ValueError(
                f"Invalid standard type: {value}. Allowed values: {allowed_standard_types}"
            )
        return value


    @field_validator("require_units")
    def check_require_units(cls, value):
        if value not in ["nM", "uM", "mM"]:
            raise ValueError(
                f"Invalid require_units: {value}. Allowed values: ['nM', 'uM', 'mM']"
            )
        return value



class PermissiveChEMBLTargetCurator(ChEMBLTargetCuratorBase):
    """
    Implement ChEMBL curation for a given protein target.

    This is a permissive curation, where we don't apply many filters to the resulting data
    """

    require_pchembl: bool = Field(True, description="Require a pChEMBL value, this will constrain, units and other values to be present")


    @model_validator(mode="after")
    def check_units_pchembl(self):
        if self.require_pchembl is True:
            if self.require_units != "nM":
                raise ValueError(
                    "If require_pchembl is True, require_units must be 'nM'."
                )

        return self

    def get_templated_query(self) -> str:
        template_str = """
        -- Get all the activity data for a given target using its ChEMBL ID.
        select
            activities.assay_id                  as assay_id,
            activities.doc_id                    as doc_id,
            activities.standard_value            as standard_value,
            molecule_hierarchy.parent_molregno   as molregno,
            compound_structures.canonical_smiles as canonical_smiles,
            compound_structures.standard_inchi_key as standard_inchi_key,
            target_dictionary.tid                as tid,
            target_dictionary.chembl_id          as target_chembl_id,
            pchembl_value                        as pchembl_value,
            molecule_dictionary.pref_name        as compound_name,
            activities.standard_type             as standard_type,
            activities.bao_endpoint              as bao_endpoint,
            assays.description                   as assay_description,
            assays.assay_organism                as assay_organism,
            assays.assay_strain                  as assay_strain,
            assays.assay_tissue                  as assay_tissue,
            assays.assay_type                    as assay_type,
            assays.assay_cell_type               as assay_cell_type,
            assays.assay_subcellular_fraction    as assay_subcellular_fraction,
            assays.variant_id                    as variant_id,
            docs.year                            as doc_year,
            docs.journal                         as doc_journal,
            docs.doi                             as doc_doi,
            docs.title                           as doc_title,
            docs.authors                         as doc_authors,
            docs.abstract                        as doc_abstract,
            docs.patent_id                       as doc_patent_id,
            docs.pubmed_id                       as doc_pubmed_id,
            docs.chembl_release_id               as doc_chembl_release_id
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join target_components ON target_dictionary.tid = target_components.tid
        join component_class ON target_components.component_id = component_class.component_id
        join docs ON activities.doc_id = docs.doc_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where target_chembl_id = '{{ target_chembl_id }}'
        and activities.data_validity_comment IS null
        {% if require_units %}and activities.standard_units = '{{ require_units }}'{% endif %}
        {% if require_pchembl %}and pchembl_value is not null{% endif %}
        {% if standard_type %}and standard_type ='{{ standard_type }}'{% endif %}
        """

        # Render the query in Python
        template = Template(template_str)

        query = template.render(
            require_units=self.require_units,
            target_chembl_id=self.chembl_target_id,
            require_pchembl=self.require_pchembl,
            standard_type=self.standard_type,
        )

        return query






class SemiQuantChEMBLTargetCurator(ChEMBLTargetCuratorBase):
    """
    Implement all ChEMBL curation for a given protein target.

    This curation includes both quantitative and semiquantitative data, i.e. activity data that is out of range of the assay.
    In terms of ChEMBL curation, this means that 'standard_type' can be '=', '<', '>', '<=', or '>='.
    pChEMBL value is explicitly NOT required.

    """



    def get_templated_query(self) -> str:

        template_str = """
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        activities.standard_relation         as standard_relation,
        activities.standard_type             as standard_type,
        activities.standard_units            as standard_units,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        compound_structures.standard_inchi_key as standard_inchi_key,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        activities.standard_type             as standard_type,
        activities.bao_endpoint              as bao_endpoint,
        assays.description                   as assay_description,
        assays.assay_organism                as assay_organism,
        assays.assay_strain                  as assay_strain,
        assays.assay_tissue                  as assay_tissue,
        assays.assay_type                    as assay_type,
        assays.assay_cell_type               as assay_cell_type,
        assays.assay_subcellular_fraction    as assay_subcellular_fraction,
        assays.variant_id                    as variant_id,
        assays.confidence_score              as confidence_score,
        docs.year                            as doc_year,
        docs.journal                         as doc_journal,
        docs.doi                             as doc_doi,
        docs.title                           as doc_title,
        docs.authors                         as doc_authors,
        docs.abstract                        as doc_abstract,
        docs.patent_id                       as doc_patent_id,
        docs.pubmed_id                       as doc_pubmed_id,
        docs.chembl_release_id               as doc_chembl_release_id
        from activities

        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join target_components ON target_dictionary.tid = target_components.tid
        join component_class ON target_components.component_id = component_class.component_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        join docs ON activities.doc_id = docs.doc_id
        where target_chembl_id = '{{ target_chembl_id }}' and
        and standard_relation in ('=', '<', '>', '<=', '>=')
        activities.data_validity_comment IS null and
        {% if require_units %}and activities.standard_units = '{{ require_units }}'{% endif %}
        {% if require_pchembl %}and pchembl_value is not null{% endif %}
        {% if standard_type %}and standard_type ='{{ standard_type }}' {% else %} and standard_type IN ('IC50', 'XC50', 'EC50', 'AC50', 'Ki', 'Kd', 'Potency') {% endif %}
        """


        # Render the query in Python
        template = Template(template_str)

        query = template.render(
            require_units=self.require_units,
            target_chembl_id=self.chembl_target_id,
            require_pchembl=self.require_pchembl,
            standard_type=self.standard_type,
        )

        return query



class MICChEMBLCurator(ChEMBLCuratorBase):
    """Curates MIC data from ChEMBL"""

    organism: Optional[str] = Field(None, description="Organism to filter the target data by.")

    include_out_of_range: bool = Field(
        False,
        description="If True, will include activities with out of range values (e.g. > 10000 nM).",
    )


    def get_templated_query(self) -> str:
        """

        Get the templated query for the MIC data to pull from the ChEMBL database.
        """
        query = """
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        activities.standard_relation         as standard_relation,
        activities.standard_type             as standard_type,
        activities.standard_units            as standard_units,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        compound_structures.standard_inchi_key as standard_inchi_key,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        activities.standard_type             as standard_type,
        assays.description                   as assay_description,
        assays.assay_organism                as assay_organism,
        assays.assay_strain                  as assay_strain,
        assays.assay_tissue                  as assay_tissue,
        assays.assay_type                    as assay_type,
        assays.assay_cell_type               as assay_cell_type,
        assays.assay_subcellular_fraction    as assay_subcellular_fraction,
        assays.variant_id                    as variant_id,
        assays.confidence_score             as confidence_score,
        docs.year                            as doc_year,
        docs.journal                         as doc_journal,
        docs.doi                             as doc_doi,
        docs.title                           as doc_title,
        docs.authors                         as doc_authors,
        docs.abstract                        as doc_abstract,
        docs.patent_id                       as doc_patent_id,
        docs.pubmed_id                       as doc_pubmed_id,
        docs.chembl_release_id               as doc_chembl_release_id
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join docs ON activities.doc_id = docs.doc_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where activities.standard_type = 'MIC' and
        target_dictionary.target_type = 'ORGANISM'
        and activities.data_validity_comment IS null
        {% if organism %}and assays.assay_organism = '{{ organism }}' {% endif %}
        {% if include_out_of_range %} and activities.standard_relation in ('=', '<', '>', '<=', '>=') {% else %} and activities.standard_relation = '=' {% endif %}
        """


        # Render the query in Python
        template = Template(query)
        query = template.render(
            organism=self.organism,
            include_out_of_range=self.include_out_of_range
        )
        return query




class HepatotoxicityChEMBLCurator(ChEMBLCuratorBase):


    def get_templated_query(self) -> str:
        """
        Get the templated query for the hepatotoxicity data to pull from the ChEMBL database.
        """
        query = """
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        activities.standard_relation         as standard_relation,
        activities.standard_type             as standard_type,
        activities.standard_units            as standard_units,
        activities.activity_comment         as activity_comment,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        compound_structures.standard_inchi_key as standard_inchi_key,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        activities.standard_type             as standard_type,
        assays.description                   as assay_description,
        assays.assay_organism                as assay_organism,
        assays.assay_strain                  as assay_strain,
        assays.assay_tissue                  as assay_tissue,
        assays.assay_type                    as assay_type,
        assays.assay_cell_type               as assay_cell_type,
        assays.assay_subcellular_fraction    as assay_subcellular_fraction,
        assays.variant_id                    as variant_id,
        assays.confidence_score             as confidence_score,
        docs.year                            as doc_year,
        docs.journal                         as doc_journal,
        docs.doi                             as doc_doi,
        docs.title                           as doc_title,
        docs.authors                         as doc_authors,
        docs.abstract                        as doc_abstract,
        docs.patent_id                       as doc_patent_id,
        docs.pubmed_id                       as doc_pubmed_id,
        docs.chembl_release_id               as doc_chembl_release_id
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join docs ON activities.doc_id = docs.doc_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where activities.standard_type like '%Hepatotoxicity%' and
        target_dictionary.target_type = 'PHENOTYPE'
        """
        # Render the query in Python
        template = Template(query)
        query = template.render()
        return query



class LogPDCurator(ChEMBLCuratorBase):
    """
    Curator for LogP/D data from ChEMBL.

    This class provides methods to curate LogP/D data from the ChEMBL database.
    """
    standard_type: str = Field(
        "LogD",
        description="Standard type to filter the LogP/D data. Default is 'LogD'"
    )

    @field_validator("standard_type")
    def check_standard_type(cls, value):
        allowed_standard_types = ["LogP", "LogD"]
        if value not in allowed_standard_types:
            raise ValueError(
                f"Invalid standard type: {value}. Allowed values: {allowed_standard_types}"
            )
        return value


    def get_templated_query(self) -> str:
        """
        Get the templated query for the LogP/D data to pull from the ChEMBL database.
        """
        query = """
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        activities.standard_relation         as standard_relation,
        activities.standard_type             as standard_type,
        activities.standard_units            as standard_units,
        activities.activity_comment         as activity_comment,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        compound_structures.standard_inchi_key as standard_inchi_key,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        activities.standard_type             as standard_type,
        assays.description                   as assay_description,
        assays.assay_organism                as assay_organism,
        assays.assay_strain                  as assay_strain,
        assays.assay_tissue                  as assay_tissue,
        assays.assay_type                    as assay_type,
        assays.assay_cell_type               as assay_cell_type,
        assays.assay_subcellular_fraction    as assay_subcellular_fraction,
        assays.variant_id                    as variant_id,
        assays.confidence_score             as confidence_score,
        assays.bao_format                   as bao_format,
        docs.year                            as doc_year,
        docs.journal                         as doc_journal,
        docs.doi                             as doc_doi,
        docs.title                           as doc_title,
        docs.authors                         as doc_authors,
        docs.abstract                        as doc_abstract,
        docs.patent_id                       as doc_patent_id,
        docs.pubmed_id                       as doc_pubmed_id,
        docs.chembl_release_id               as doc_chembl_release_id
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join docs ON activities.doc_id = docs.doc_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where activities.standard_type = '{{ standard_type }}' and
        bao_format = 'BAO_0000100'
        -- BAO_0000100 is the format for small molecule physicochemical properties
        """


        # Render the query in Python
        template = Template(query)
        query = template.render(standard_type=self.standard_type)
        return query


class pKaCurator(ChEMBLCuratorBase):
    """
    Curator for pKa data from ChEMBL.

    This class provides methods to curate pKa data from the ChEMBL database.
    """



    def get_templated_query(self) -> str:
        """
        Get the templated query for the LogP/D data to pull from the ChEMBL database.
        """
        query = """
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        activities.standard_relation         as standard_relation,
        activities.standard_type             as standard_type,
        activities.standard_units            as standard_units,
        activities.activity_comment         as activity_comment,
        activities.bao_endpoint              as bao_endpoint,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        compound_structures.standard_inchi_key as standard_inchi_key,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        activities.standard_type             as standard_type,
        assays.description                   as assay_description,
        assays.assay_organism                as assay_organism,
        assays.assay_strain                  as assay_strain,
        assays.assay_tissue                  as assay_tissue,
        assays.assay_type                    as assay_type,
        assays.assay_cell_type               as assay_cell_type,
        assays.assay_subcellular_fraction    as assay_subcellular_fraction,
        assays.variant_id                    as variant_id,
        assays.confidence_score             as confidence_score,
        assays.bao_format                   as bao_format,
        docs.year                            as doc_year,
        docs.journal                         as doc_journal,
        docs.doi                             as doc_doi,
        docs.title                           as doc_title,
        docs.authors                         as doc_authors,
        docs.abstract                        as doc_abstract,
        docs.patent_id                       as doc_patent_id,
        docs.pubmed_id                       as doc_pubmed_id,
        docs.chembl_release_id               as doc_chembl_release_id
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join docs ON activities.doc_id = docs.doc_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where activities.standard_type = 'pKa' and
        bao_format = 'BAO_0000100'
        -- BAO_0000100 is the format for small molecule physicochemical properties
        """


        # Render the query in Python

        template = Template(query)
        query = template.render()
        return query

class MicrosomalChEMBLCurator(ChEMBLCuratorBase):
    """
    Curator for microsomal  data from ChEMBL.

    This class provides methods to curate microsomal stability data from the ChEMBL database.
    """
    organism: Optional[str] = Field(
        None, description="Organism to filter the microsomal stability data by."
    )
    require_units: str = Field(
        "mL.min-1.g-1", description="Required units for the microsomal stability data."
    )

    standard_type: str = Field(
        "CL",
        description="Standard type to filter the microsomal stability data. Default is 'CL' (Clearance), other options include 'T1/2' (T1/2, half-life)."
    )

    @field_validator("standard_type")
    def check_standard_type(cls, value):
        allowed_standard_types = ["CL", "T1/2"]
        if value not in allowed_standard_types:
            raise ValueError(
                f"Invalid standard type: {value}. Allowed values: {allowed_standard_types}"
            )
        return value

    def get_templated_query(self) -> str:
        """
        Get the templated query for the microsomal stability data to pull from the ChEMBL database.
        """
        query = """
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        activities.standard_relation         as standard_relation,
        activities.standard_type             as standard_type,
        activities.standard_units            as standard_units,
        activities.activity_comment         as activity_comment,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        compound_structures.standard_inchi_key as standard_inchi_key,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        activities.standard_type             as standard_type,
        assays.description                   as assay_description,
        assays.assay_organism                as assay_organism,
        assays.assay_strain                  as assay_strain,
        assays.assay_tissue                  as assay_tissue,
        assays.assay_type                    as assay_type,
        assays.assay_cell_type               as assay_cell_type,
        assays.assay_subcellular_fraction    as assay_subcellular_fraction,
        assays.variant_id                    as variant_id,
        assays.confidence_score             as confidence_score,
        assays.bao_format                   as bao_format,
        docs.year                            as doc_year,
        docs.journal                         as doc_journal,
        docs.doi                             as doc_doi,
        docs.title                           as doc_title,
        docs.authors                         as doc_authors,
        docs.abstract                        as doc_abstract,
        docs.patent_id                       as doc_patent_id,
        docs.pubmed_id                       as doc_pubmed_id,
        docs.chembl_release_id               as doc_chembl_release_id
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join docs ON activities.doc_id = docs.doc_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where activities.standard_type = '{{ standard_type }}' and
        bao_format = 'BAO_0000251'
        -- BAO_0000251 is the format for microsomal assays
        {% if organism %}and assays.assay_organism = '{{ organism }}' {% endif %}
        {% if require_units %}and activities.standard_units = '{{ require_units }}' {% endif %}
        """

        # Render the query in Python
        template = Template(query)
        query = template.render(organism=self.organism, standard_type=self.standard_type, require_units=self.require_units)
        return query


class PPBChEMBLCurator(ChEMBLCuratorBase):
    """
    Curator for plasma protein binding (PPB) data from ChEMBL.

    This class provides methods to curate plasma protein binding data from the ChEMBL database.
    """
    organism: Optional[str] = Field(
        None, description="Organism to filter the plasma protein binding data by."
    )

    def get_templated_query(self) -> str:
        """
        Get the templated query for the microsomal stability data to pull from the ChEMBL database.
        """
        query = """
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        activities.standard_relation         as standard_relation,
        activities.standard_type             as standard_type,
        activities.standard_units            as standard_units,
        activities.activity_comment         as activity_comment,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        compound_structures.standard_inchi_key as standard_inchi_key,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        activities.standard_type             as standard_type,
        assays.description                   as assay_description,
        assays.assay_organism                as assay_organism,
        assays.assay_strain                  as assay_strain,
        assays.assay_tissue                  as assay_tissue,
        assays.assay_type                    as assay_type,
        assays.assay_cell_type               as assay_cell_type,
        assays.assay_subcellular_fraction    as assay_subcellular_fraction,
        assays.variant_id                    as variant_id,
        assays.confidence_score             as confidence_score,
        assays.bao_format                   as bao_format,
        docs.year                            as doc_year,
        docs.journal                         as doc_journal,
        docs.doi                             as doc_doi,
        docs.title                           as doc_title,
        docs.authors                         as doc_authors,
        docs.abstract                        as doc_abstract,
        docs.patent_id                       as doc_patent_id,
        docs.pubmed_id                       as doc_pubmed_id,
        docs.chembl_release_id               as doc_chembl_release_id
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join docs ON activities.doc_id = docs.doc_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where activities.standard_type = 'PPB' and
        activities.standard_units = '%' and
        assay_type in ('ADMET') and
        activities.bao_format = 'BAO_0000366'
        -- BAO_0000366 is the format for cell-free assays
        {% if organism %}and assays.assay_organism = '{{ organism }}' {% endif %}
        """

        # Render the query in Python
        template = Template(query)
        query = template.render(organism=self.organism)
        return query
