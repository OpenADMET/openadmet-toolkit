import pandas as pd
from rdkit import Chem
import seaborn as sns
from tqdm import tqdm
import numpy as np

from rdkit.rdBase import BlockLogs
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
from rdkit_funcs import standardize_smiles, smiles_to_inchikey

class ChEMBLProcessing(BaseModel):
    """
    Class to handle processing data from a csv downloaded
    from ChEMBL

    """
    csv_path: Path = Field(..., description="Path to the ChEMBL csv")
    
    def __init__(self, **data):
        super().__init__(**data)
        self.chemblData = self.read_csv(self.csv_path)
        self.keep_cols = ["CANONICAL_SMILES", "INCHIKEY", "pChEMBL mean", "pChEMBL std", "Molecule Name", "assay_count", "Action Type"]
        self.standardize_smiles_and_convert()

    @classmethod
    def read_csv():
        return(pd.read_csv("ChEMBL/ChEMBL_CYP1A2_activities.csv", sep=";"))

    @classmethod
    def standardize_smiles_and_convert(self):
        with BlockLogs():
            self.chemblData["CANONICAL_SMILES"] = self.chemblData["Smiles"].progress_apply(lambda x: standardize_smiles(x))
        with BlockLogs():
            self.chemblData["INCHIKEY"] = self.chemblData["CANONICAL_SMILES"].progress_apply(lambda x: smiles_to_inchikey(x))
        self.chemblData.dropna(subset="INCHIKEY", inplace=True)

    @classmethod
    def select_quality_data_inhibition(self, N: int = 10, pchembl_thresh: float = 5.0, L: int = 1, save=True):
        better_assay = self.chemblData[
            (self.chemblData['Standard Type'] == 'IC50') |
            (self.chemblData['Standard Type'] == 'AC50') |
            (self.chemblData['Standard Type'] == 'pIC50') |
            (self.chemblData['Standard Type'] == 'XC50') |
            (self.chemblData['Standard Type'] == 'EC50') | 
            (self.chemblData['Standard Type'] == 'Ki') |
            (self.chemblData['Standard Type'] == 'Potency')
        ]
        better_units = better_assay[better_assay['Standard Units'] == "nM"]
        combined = self.get_num_compounds_per_assay(better_units)
        
        more_than_N_compounds = self.get_more_than_N_compounds(combined)
        assays = more_than_N_compounds["Assay ChEMBL ID"].nunique()
        num_assays_per_compound_df = self.get_num_assays_per_compound(more_than_N_compounds)
        combined_2 = more_than_N_compounds.join(num_assays_per_compound_df, on="INCHIKEY")
        combined_2.sort_values("assay_count", ascending=False, inplace=True)
        combined_2["assay_count"] = combined_2["assay_count"].astype(int)
        
        compound_grouped_mean = combined_2.groupby("INCHIKEY")["pChEMBL Value"].mean()
        compound_grouped_mean.reset_index()

        cgm = compound_grouped_mean.reset_index(name="pChEMBL mean")
        cgm = cgm.set_index("INCHIKEY")
        combined_3 = combined_2.join(cgm, on="INCHIKEY")

        compound_grouped_std = combined_2.groupby("INCHIKEY")["pChEMBL Value"].std()

        cgstd = compound_grouped_std.reset_index(name="pChEMBL std")
        cgstd = cgstd.set_index("INCHIKEY")
        combined_4 =  combined_3.join(cgstd, on="INCHIKEY")

        # get active compounds
        # defined as compounds above pChEMBL value specified (default 5.0)
        active = combined_4[combined_4["pChEMBL mean"] >= pchembl_thresh]
        clean_deduped = self.clean_and_dedupe_actives(active, save, inhibition=True)
        self.more_than_L_assays(clean_deduped, L)

    def select_quality_data_reactivity(self, save):
        substrates = self.chemblData[self.chemblData["Action Type"] == "SUBSTRATE"]
        self.clean_and_dedupe_activities(substrates, save, inhibition=False)

    def more_than_L_assays(self, clean_deduped, L=1, save=True):
        more_than_eq_L_assay = clean_deduped[clean_deduped["appears_in_N_ChEMBL_assays"] >= L]
        more_than_eq_L_assay.to_csv("processed/chembl_active_selected.csv", index=False)
        return(more_than_eq_L_assay.INCHIKEY.nunique())

    def clean_and_dedupe_actives(self, active, save=True, inhibition=True):
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
        clean_deduped["active"] = True
        if save:    
            clean_deduped.to_csv("processed/chembl_active.csv", index=False)
        return(clean_deduped)

    def get_more_than_N_compounds(self, combined):
        more_than_N_compounds = combined[combined["molecule_count"] > N]
        more_than_N_compounds.INCHIKEY = more_than_N_compounds.INCHIKEY.astype(str)
        return(more_than_N_compounds["Assay ChEMBL ID"].nunique())

    def get_num_assays_per_compound(self, more_than_N_compounds):
        num_assays_per_compound_df = more_than_N_compounds.groupby(["INCHIKEY"])["Assay ChEMBL ID"].size().reset_index(name="assay_count")
        return(num_assays_per_compound_df.set_index("INCHIKEY"))

    def get_num_compounds_per_assay(self, better_units):
        num_compounds_per_assay = better_units.groupby("Assay ChEMBL ID")["Molecule ChEMBL ID"].nunique()
        num_compounds_per_assay_df = pd.DataFrame(num_compounds_per_assay)
        num_compounds_per_assay_df.rename(columns={"Molecule ChEMBL ID": "molecule_count"}, inplace=True)
        return(better_units.join(num_compounds_per_assay_df, on="Assay ChEMBL ID"))

    def aggregate_activity(self, combined_2):
        compound_grouped_mean = combined_2.groupby("INCHIKEY")["pChEMBL Value"].mean()
        return(compound_grouped_mean.reset_index())

    def get_num_assays_per_compound(self, more_than_N_compounds):
        num_assays_per_compound_df = more_than_N_compounds.groupby(["INCHIKEY"])["Assay ChEMBL ID"].size().reset_index(name="assay_count")
        return(num_assays_per_compound_df.set_index("INCHIKEY"))
