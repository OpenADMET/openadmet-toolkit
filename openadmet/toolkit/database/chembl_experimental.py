import abc
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, Union

import chembl_downloader
import datamol as dm
import duckdb
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import warnings

from openadmet.toolkit.chemoinformatics.rdkit_funcs import canonical_smiles, smiles_to_inchikey

from openadmet.toolkit.database.chembl import ChemblConnector, ChEMBLTargetCuratorBase


# this implements curation as described in the landrum paper
# https://pubs.acs.org/doi/10.1021/acs.jcim.4c00049
# with accompanying code here: https://github.com/rinikerlab/overlapping_assays/tree/main
# however I am not sure the implementation is correct, so use with caution, relvant warnings are raised



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
        self, return_as: str = "df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """
        Get the high quality assays for a given target using its ChEMBL ID.

        Parameters
        ----------
        return_as: str
            Return the data as a DataFrame or a DuckDB relation.
        """
        warnings.warn("This curation is experimental and may not be correct.",
                      UserWarning)
        # first we get the assays for the target using a resonable level of curation.
        query = f"""
            drop table if exists temp_assay_data;
            create temporary table temp_assay_data as
            select assay_id,assays.chembl_id assay_chembl_id,description,tid,targets.chembl_id target_chembl_id,\
            count(distinct(molregno)) molcount,pref_name,assays.doc_id doc_id,docs.year doc_date,variant_id, \
            assays.confidence_score, standard_type \
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
        self, return_as: str = "df"
    ) -> Union[pd.DataFrame, duckdb.DuckDBPyRelation]:
        """
        Get the high quality activity data for a given target using its ChEMBL ID.
        """
        warnings.warn("This curation is experimental and may not be correct.",
                      UserWarning)
        hq_assays = self.get_high_quality_assays_for_target(return_as="duckdb")
        # get all the activity data for the target
        query = f"""
        select
        activities.assay_id                  as assay_id,
        activities.doc_id                    as doc_id,
        activities.standard_value            as standard_value,
        molecule_hierarchy.parent_molregno   as molregno,
        compound_structures.canonical_smiles as canonical_smiles,
        target_dictionary.tid                as tid,
        target_dictionary.chembl_id          as target_chembl_id,
        pchembl_value                        as pchembl_value,
        molecule_dictionary.pref_name        as compound_name,
        from activities
        join assays ON activities.assay_id = assays.assay_id
        join target_dictionary ON assays.tid = target_dictionary.tid
        join target_components ON target_dictionary.tid = target_components.tid
        join component_class ON target_components.component_id = component_class.component_id
        join molecule_dictionary ON activities.molregno = molecule_dictionary.molregno
        join molecule_hierarchy ON molecule_dictionary.molregno = molecule_hierarchy.molregno
        join compound_structures ON molecule_hierarchy.parent_molregno = compound_structures.molregno
        where activities.standard_units = 'nM' and
        pchembl_value is not null and
        activities.data_validity_comment IS null and
        activities.standard_relation = '=' and
        activities.potential_duplicate = 0 and
        assays.confidence_score >= 8 and
        target_dictionary.target_type = 'SINGLE PROTEIN' and
        target_chembl_id = '{self.chembl_target_id}'
        """

        all_data = self._chembl_connector.query(query, return_as="duckdb")

        # find intersection of high quality assays and all data
        hq_data = hq_assays.join(all_data, "assay_id, tid, target_chembl_id, doc_id")

        if return_as == "df":
            return hq_data.to_df()
        else:
            return hq_data

    def aggregate_activity_data_by_compound(self, canonicalise=False) -> pd.DataFrame:
        """
        Get the high quality activity data for a given target using its ChEMBL ID, grouped by compound.
        If canonicalise is True, the SMILES will be canonicalised and the InChIKey calculated and
        aggregation will be done on the canonicalised SMILES and InChIKey.

        Parameters
        ----------
        canonicalise: bool
            Canonicalise the SMILES and calculate the InChIKey

        Returns
        -------
        pd.DataFrame
            DataFrame containing the aggregated activity data.
        """
        warnings.warn("This curation is experimental and may not be correct.",
                      UserWarning)
        hq_data = self.get_activity_data(return_as="df")
        if canonicalise:
            with dm.without_rdkit_log():
                hq_data["OPENADMET_SMILES"] = hq_data[
                    "canonical_smiles"
                ].progress_apply(lambda x: canonical_smiles(x))
                hq_data["OPENADMET_INCHIKEY"] = hq_data[
                    "OPENADMET_SMILES"
                ].progress_apply(lambda x: smiles_to_inchikey(x))
                data = hq_data.groupby(
                    [
                        "OPENADMET_SMILES",
                        "OPENADMET_INCHIKEY",
                    ]
                ).agg(
                    {
                        "assay_id": "count",
                        "standard_value": ["mean", "std"],
                        "pchembl_value": ["mean", "std"],
                    }
                )
        else:
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
        data.sort_values("assay_id_count", ascending=False, inplace=True)
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
        select distinct(variant_id), tid, description,  targets.chembl_id as target_chembl_id, \
        assay_id, assays.chembl_id assay_chembl_id \
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
