import pandas as pd
from rdkit import Chem
import seaborn as sns
from tqdm import tqdm
import numpy as np

from rdkit.rdBase import BlockLogs
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from rdkit_funcs import standardize_smiles, smiles_to_inchikey

class CSVProcessing(BaseModel):
    """
    Class to handle processing data from a csv downloaded
    """
    csv_path: Path = Field(..., description="Path to the ChEMBL csv")

    @staticmethod
    def read_csv(csv_path, sep=','):
        return(pd.read_csv(csv_path, sep))

    @staticmethod
    def standardize_smiles_and_convert(data):
        with BlockLogs():
            data["CANONICAL_SMILES"] = data["Smiles"].progress_apply(lambda x: standardize_smiles(x))
        with BlockLogs():
            data["INCHIKEY"] = data["CANONICAL_SMILES"].progress_apply(lambda x: smiles_to_inchikey(x))
        return(data.dropna(subset="INCHIKEY", inplace=True))
    
class ChEMBLProcessing(CSVProcessing):
    """
    Class to handle processing data from a csv downloaded
    from ChEMBL

    """
    def __init__(self, **data):
        super().__init__(**data)
        self.data = self.read_csv(self.csv_path, ";")
        self.keep_cols = ["CANONICAL_SMILES", "INCHIKEY", "pChEMBL mean", "pChEMBL std", "Molecule Name", "assay_count", "Action Type"]
        self.data = self.standardize_smiles_and_convert(self.data)

    def select_quality_data_inhibition(self, min_compound_num=None, pchembl_thresh=None, min_assay_num=None, save_as=None):
        better_assay = self.data[
            (self.data['Standard Type'] == 'IC50') |
            (self.data['Standard Type'] == 'AC50') |
            (self.data['Standard Type'] == 'pIC50') |
            (self.data['Standard Type'] == 'XC50') |
            (self.data['Standard Type'] == 'EC50') | 
            (self.data['Standard Type'] == 'Ki') |
            (self.data['Standard Type'] == 'Potency')
        ]
        better_units = better_assay[better_assay['Standard Units'] == "nM"]
        combined_df = self.get_num_compounds_per_assay(better_units)
        
        more_than_N_compounds = self.get_more_than_N_compounds(combined_df, min_compound_num)
        assays = more_than_N_compounds["Assay ChEMBL ID"].nunique()
        # on the following line, should more_than_N_compounds be assays?
        num_assays_per_compound_df = self.get_num_assays_per_compound(assays)
        combined_df = more_than_N_compounds.join(num_assays_per_compound_df, on="INCHIKEY")
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
        combined_df =  combined_df.join(cgstd, on="INCHIKEY")

        # get active compounds
        # defined as compounds above pChEMBL value specified 
        # (default 5.0 from https://greglandrum.github.io/rdkit-blog/posts/2023-06-12-overlapping-ic50-assays1.html)
        if pchembl_thresh != None:
            active = combined_df[combined_df["pChEMBL mean"] >= pchembl_thresh]
        else:
            active = combined_df.copy()
        clean_deduped = self.clean_and_dedupe_actives(active, save_as, inhibition=True)
        if min_assay_num != None:
            return(self.more_than_L_assays(clean_deduped, min_assay_num))
        else:
            return(clean_deduped)

    def select_quality_data_reactivity(self, save_as):
        substrates = self.data[self.data["Action Type"] == "SUBSTRATE"]
        return(self.clean_and_dedupe_activities(substrates, save_as, inhibition=False))

    def more_than_L_assays(self, clean_deduped, min_assay_num, save_as=None):
        if min_assay_num != None:
            more_than_eq_L_assay = clean_deduped[clean_deduped["appears_in_N_ChEMBL_assays"] >= min_assay_num]
        else:
            more_than_eq_L_assay = clean_deduped.copy()
        if save_as is not None:
            more_than_eq_L_assay.to_csv(save_as, index=False)
        return(more_than_eq_L_assay.INCHIKEY.nunique())

    def clean_and_dedupe_actives(self, active, save_as=None, inhibition=True):
        clean_active = active[self.keep_cols]
        clean_active.rename(columns={"assay_count":"appears_in_N_ChEMBL_assays", "Molecule Name": "common_name", "Action Type": "action_type"}, inplace=True)
        clean_active_sorted = clean_active.sort_values(["common_name", "action_type"], ascending=[False, False]) # keep the ones with names if possible
        clean_deduped = clean_active_sorted.drop_duplicates(subset="INCHIKEY", keep="first")
        if inhibition:
            clean_deduped = clean_deduped.sort_values("appears_in_N_ChEMBL_assays", ascending=False)
            clean_deduped["action_type"] = clean_deduped["action_type"].apply(lambda x: x.lower() if isinstance(x, str) else x)
        else:
            clean_deduped["action_type"] = "substrate"
        clean_deduped["dataset"] = "ChEMBL_curated"
        if save_as is not None:    
            clean_deduped.to_csv(save_as, index=False)
        return(clean_deduped)

    def get_more_than_N_compounds(self, combined, min_compound_num):
        if min_compound_num != None:
            more_than_N_compounds = combined[combined["molecule_count"] > min_compound_num]
            more_than_N_compounds.INCHIKEY = more_than_N_compounds.INCHIKEY.astype(str)
            return(more_than_N_compounds["Assay ChEMBL ID"].nunique())
        else:
            return(combined)

    def get_num_assays_per_compound(self, more_than_N_compounds):
        num_assays_per_compound_df = more_than_N_compounds.groupby(["INCHIKEY"])["Assay ChEMBL ID"].size().reset_index(name="assay_count")
        return(num_assays_per_compound_df.set_index("INCHIKEY"))

    def get_num_compounds_per_assay(self, better_units):
        num_compounds_per_assay = better_units.groupby("Assay ChEMBL ID")["Molecule ChEMBL ID"].nunique()
        num_compounds_per_assay_df = pd.DataFrame(num_compounds_per_assay)
        num_compounds_per_assay_df.rename(columns={"Molecule ChEMBL ID": "molecule_count"}, inplace=True)
        return(better_units.join(num_compounds_per_assay_df, on="Assay ChEMBL ID"))

    def aggregate_activity(self, combined_df):
        compound_grouped_mean = combined_df.groupby("INCHIKEY")["pChEMBL Value"].mean()
        return(compound_grouped_mean.reset_index())

class PubChemProcessing(CSVProcessing):
    """
    Class to handle processing data from a csv downloaded
    from PubChem

    """
    def __init__(self, **data):
        super().__init__(**data)
        self.data = self.read_csv(self.csv_path)
        self.keep_cols = ["CANONICAL_SMILES", "INCHIKEY", "PUBCHEM_ACTIVITY_OUTCOME", "PUBCHEM_CID"]
        self.delete_metadata_rows()
        self.data = self.data.dropna(subset="PUBCHEM_CID")
        self.data["PUBCHEM_SID"] = self.data["PUBCHEM_SID"].astype(int)
        self.data["PUBCHEM_CID"] = self.data["PUBCHEM_CID"].astype(int)
        self.data = self.standardize_smiles_and_convert(self.data)   
        self.data.dropna(subset="INCHIKEY")

    @classmethod
    def delete_metadata_rows(self):
        to_del = 0
        for index, row in self.data.iterrows():
            if index == 0:
                continue
            elif Chem.MolFromSmiles(row['PUBCHEM_EXT_DATASOURCE_SMILES']) is not None:
                to_del += 1
            else:
                break
        self.data = self.data.drop(labels=list(range(0, to_del)), axis=0).reset_index(drop=True)
        
    def clean_data_inhibition(self, aid, data_type, save_as=None):
        clean = self.data[self.keep_cols]
        clean["dataset"] = aid
        clean["data_type"] = data_type
        clean["active"] = clean["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"
        clean["common_name"] = pd.NA
        clean["action_type"] = "inhibitor"
        if save_as is not None:
            clean.to_csv(save_as, index=False)
