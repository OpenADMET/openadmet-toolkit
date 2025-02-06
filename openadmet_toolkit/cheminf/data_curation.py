from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, Field, field_validator
from rdkit import Chem

from openadmet_toolkit.cheminf.rdkit_funcs import canonical_smiles, smiles_to_inchikey


class CSVProcessing(BaseModel):
    """
    Class to handle processing data from a csv downloaded
    """

    @staticmethod
    def read_csv(csv_path, sep=","):
        return pd.read_csv(csv_path, sep=sep)

    @staticmethod
    def standardize_smiles_and_convert(data):
        data["CANONICAL_SMILES"] = data["Smiles"].apply(lambda x: canonical_smiles(x))
        data["INCHIKEY"] = data["CANONICAL_SMILES"].apply(
            lambda x: smiles_to_inchikey(x)
        )
        data.dropna(subset="INCHIKEY", inplace=True)
        return data


class ChEMBLProcessing(CSVProcessing):
    """
    Class to handle processing data from a csv downloaded
    from ChEMBL

    """

    inhib: bool = Field(default=False)
    react: bool = Field(default=False)
    min_compound_num: int = Field(default=1)
    pchembl_thresh: float = Field(default=5.0)
    min_assay_num: int = Field(default=1)
    save_as: str = Field(default=None)
    keep_cols_inhib: list[str] = Field(
        default=[
            "Smiles",
            "CANONICAL_SMILES",
            "INCHIKEY",
            "pChEMBL mean",
            "pChEMBL std",
            "Molecule Name",
            "assay_count",
            "Action Type",
        ]
    )
    keep_cols_react: list[str] = Field(
        default=[
            "Smiles",
            "CANONICAL_SMILES",
            "INCHIKEY",
            "Molecule Name",
            "Action Type",
        ]
    )

    def process(self, path):
        data = self.read_csv(path, ";")
        data = self.standardize_smiles_and_convert(data)
        if self.inhib:
            df = self.select_quality_data_inhibition(
                data,
                self.min_compound_num,
                self.pchembl_thresh,
                self.min_assay_num,
                self.save_as,
            )
        elif self.react:
            df = self.select_quality_data_reactivity(data, self.save_as)
        else:
            raise ValueError("Must specify either inhib or react as True.")
        return df

    def select_quality_data_inhibition(
        self,
        data,
        min_compound_num=None,
        pchembl_thresh=None,
        min_assay_num=None,
        save_as=None,
    ):
        better_assay = data[
            (data["Standard Type"] == "IC50")
            | (data["Standard Type"] == "AC50")
            | (data["Standard Type"] == "pIC50")
            | (data["Standard Type"] == "XC50")
            | (data["Standard Type"] == "EC50")
            | (data["Standard Type"] == "Ki")
            | (data["Standard Type"] == "Potency")
        ]
        better_units = better_assay[better_assay["Standard Units"] == "nM"]
        num_compounds_per_assay_df = self.get_num_compounds_per_assay(better_units)
        combined_df = better_units.join(
            num_compounds_per_assay_df, on="Assay ChEMBL ID"
        )

        more_than_N_compounds = self.get_more_than_N_compounds(
            combined_df, min_compound_num
        )
        num_assays_per_compound_df = self.get_num_assays_per_compound(
            more_than_N_compounds
        )

        more_than_N_compounds.INCHIKEY = more_than_N_compounds.INCHIKEY.astype(str)
        num_assays_per_compound_df.INCHIKEY = (
            num_assays_per_compound_df.INCHIKEY.astype(str)
        )

        combined_df = more_than_N_compounds.merge(
            num_assays_per_compound_df, on="INCHIKEY"
        )
        combined_df.sort_values("assay_count", ascending=False, inplace=True)
        combined_df["assay_count"] = combined_df["assay_count"].astype(int)

        compound_grouped_mean = combined_df.groupby("INCHIKEY")["pChEMBL Value"].mean()
        compound_grouped_mean.reset_index()

        cgm = compound_grouped_mean.reset_index(name="pChEMBL mean")
        cgm = cgm.set_index("INCHIKEY")
        combined_df = combined_df.join(cgm, on="INCHIKEY")

        compound_grouped_std = combined_df.groupby("INCHIKEY")["pChEMBL Value"].std()

        cgstd = compound_grouped_std.reset_index(name="pChEMBL std")
        cgstd = cgstd.set_index("INCHIKEY")
        combined_df = combined_df.join(cgstd, on="INCHIKEY")

        # get active compounds
        # defined as compounds above pChEMBL value specified (default 5.0)
        if pchembl_thresh != None:
            active = combined_df[combined_df["pChEMBL mean"] >= pchembl_thresh]
        else:
            active = combined_df.copy()
        clean_deduped = self.clean_and_dedupe_actives(active, save_as)
        if min_assay_num != None:
            return self.more_than_L_assays(clean_deduped, min_assay_num)
        else:
            return clean_deduped

    def select_quality_data_reactivity(self, data, save_as):
        substrates = data[data["Action Type"] == "SUBSTRATE"]
        substrates = self.clean_and_dedupe_actives(substrates, save_as)
        return substrates

    def more_than_L_assays(self, clean_deduped, min_assay_num, save_as=None):
        if min_assay_num != None:
            more_than_eq_L_assay = clean_deduped[
                clean_deduped["appears_in_N_ChEMBL_assays"] >= min_assay_num
            ]
        else:
            more_than_eq_L_assay = clean_deduped.copy()
        if save_as is not None:
            more_than_eq_L_assay.to_csv(save_as, index=False)
        return more_than_eq_L_assay

    def clean_and_dedupe_actives(self, active, save_as=None):
        if self.inhib:
            clean_active = active[self.keep_cols_inhib]
            clean_active.rename(
                columns={
                    "assay_count": "appears_in_N_ChEMBL_assays",
                    "Molecule Name": "common_name",
                    "Action Type": "action_type",
                },
                inplace=True,
            )
        else:
            clean_active = active[self.keep_cols_react]
            clean_active.rename(
                columns={
                    "Molecule Name": "common_name",
                    "Action Type": "action_type",
                },
                inplace=True,
            )
        clean_active_sorted = clean_active.sort_values(
            ["common_name", "action_type"], ascending=[False, False]
        )  # keep the ones with names if possible
        clean_deduped = clean_active_sorted.drop_duplicates(
            subset="INCHIKEY", keep="first"
        )
        if self.inhib:
            clean_deduped = clean_deduped.sort_values(
                "appears_in_N_ChEMBL_assays", ascending=False
            )
            clean_deduped["action_type"] = clean_deduped["action_type"].apply(
                lambda x: x.lower() if isinstance(x, str) else x
            )
        else:
            clean_deduped["action_type"] = "substrate"
        clean_deduped["dataset"] = "ChEMBL_curated"
        if save_as is not None:
            clean_deduped.to_csv(save_as, index=False)
        return clean_deduped

    def get_more_than_N_compounds(self, combined, min_compound_num):
        if min_compound_num != None:
            more_than_N_compounds = combined[
                combined["molecule_count"] > min_compound_num
            ]
            return more_than_N_compounds
        else:
            return combined

    def get_num_assays_per_compound(self, more_than_N_compounds):
        num_assays_per_compound_df = (
            more_than_N_compounds.groupby(["INCHIKEY"])["Assay ChEMBL ID"]
            .size()
            .reset_index(name="assay_count")
        )
        num_assays_per_compound_df.set_index("INCHIKEY")
        return num_assays_per_compound_df

    def get_num_compounds_per_assay(self, better_units):
        num_compounds_per_assay = better_units.groupby("Assay ChEMBL ID")[
            "Molecule ChEMBL ID"
        ].nunique()
        num_compounds_per_assay_df = pd.DataFrame(num_compounds_per_assay)
        num_compounds_per_assay_df.rename(
            columns={"Molecule ChEMBL ID": "molecule_count"}, inplace=True
        )
        return num_compounds_per_assay_df

    def aggregate_activity(self, combined_df):
        compound_grouped_mean = combined_df.groupby("INCHIKEY")["pChEMBL Value"].mean()
        compound_grouped_mean.reset_index()
        return compound_grouped_mean


class PubChemProcessing(CSVProcessing):
    """
    Class to handle processing data from a csv downloaded
    from PubChem

    """

    inhib: bool = Field(default=False)
    react: bool = Field(default=False)
    keep_cols: list[str] = Field(
        default=[
            "Smiles",
            "CANONICAL_SMILES",
            "INCHIKEY",
            "PUBCHEM_ACTIVITY_OUTCOME",
            "PUBCHEM_CID",
        ]
    )

    def process(self, aid):
        if self.inhib:
            data = self.read_csv(self.csv_path)
            self.delete_metadata_rows(data)
            data = data.dropna(subset="PUBCHEM_CID")
            data["PUBCHEM_SID"] = data["PUBCHEM_SID"].astype(int)
            data["PUBCHEM_CID"] = data["PUBCHEM_CID"].astype(int)
            data = self.standardize_smiles_and_convert(data)
            data.dropna(subset="INCHIKEY")
            clean = self.clean_data_inhibition()
        elif self.react:
            raise NotImplementedError(
                "Reactivity processing not implemented for PubChem"
            )
        else:
            raise ValueError("Must specify either inhib or react as True.")

    @classmethod
    def delete_metadata_rows(self, data):
        to_del = 0
        for index, row in data.iterrows():
            if index == 0:
                continue
            elif Chem.MolFromSmiles(row["PUBCHEM_EXT_DATASOURCE_SMILES"]) is not None:
                to_del += 1
            else:
                break
        data = data.drop(labels=list(range(0, to_del)), axis=0).reset_index(drop=True)

    def clean_data_inhibition(self, data, aid, data_type, save_as=None):
        clean = data[self.keep_cols]
        clean["dataset"] = aid
        clean["data_type"] = data_type
        clean["active"] = clean["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"
        clean["common_name"] = pd.NA
        clean["action_type"] = "inhibitor"
        if save_as is not None:
            clean.to_csv(save_as, index=False)
        return clean
