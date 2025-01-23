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
    def select_quality_data(self, N: int = 10):
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
        num_compounds_per_assay = better_units.groupby("Assay ChEMBL ID")["Molecule ChEMBL ID"].nunique()
        num_compounds_per_assay_df = pd.DataFrame(num_compounds_per_assay)
        num_compounds_per_assay_df.rename(columns={"Molecule ChEMBL ID": "molecule_count"}, inplace=True)
        combined = better_units.join(num_compounds_per_assay_df, on="Assay ChEMBL ID")
        combined["molecule_count"].unique()
        more_than_N_compounds = combined[combined["molecule_count"] > N]
        more_than_N_compounds.INCHIKEY = more_than_N_compounds.INCHIKEY.astype(str)
        assays = more_than_N_compounds["Assay ChEMBL ID"].nunique()
        more_than_N_compounds["Molecule ChEMBL ID"].nunique()
        num_assays_per_compound_df = more_than_N_compounds.groupby(["INCHIKEY"])["Assay ChEMBL ID"].size().reset_index(name="assay_count")
        num_assays_per_compound_df = num_assays_per_compound_df.set_index("INCHIKEY")
        combined_2 = more_than_N_compounds.join(num_assays_per_compound_df, on="INCHIKEY")
        combined_2.sort_values("assay_count", ascending=False, inplace=True)
        combined_2["assay_count"] = combined_2["assay_count"].astype(int)

    def aggregate_activity(self):
        pass

    def number_of_assay_appears(self):
        pass