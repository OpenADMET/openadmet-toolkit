from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Union
from pathlib import Path
import pandas as pd

class ReactionSMART(BaseModel):
    reaction: str = Field(..., title="Reaction SMARTS", description="The reaction SMARTS string")
    name: str = Field(None, title="Name", description="The name of the reaction")

    def reactants(self): 
        reactant_side = self.reaction.split(">>")[0]
        # split into components on the "." token and return the list
        return reactant_side.split(".")

    def products(self):
        product_side = self.reaction.split(">>")[1]
        # split into components on the "." token and return the list
        return product_side.split(".")
    


class BuildingBlockCatalouge(BaseModel):
    building_block_csv: Union[str,Path] = Field(..., title="CSV files", description="The CSV file containing the building block information")
    smiles_column: str = Field("SMILES", title="SMILES column", description="The name of the column containing the SMILES strings")
    inchikey_column: str = Field("INCHIKEY", title="INCHIKEY column", description="The name of the column containing the INCHIKEY strings")

    @field_validator("building_block_csvs")
    def check_csvs(cls, values):
        for csv in values:
            if not Path(csv).exists():
                raise ValueError(f"File {csv} does not exist")
        return values
    
    @model_validator
    def check_columns(cls, values):
        for csv in values.building_block_csv:
            df = pd.read_csv(csv)
            if values.smiles_column not in df.columns:
                raise ValueError(f"Column {values.smiles_column} not in {csv}")
            if values.inchikey_column not in df.columns:
                raise ValueError(f"Column {values.inchikey_column} not in {csv}")
        return values

    


    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.building_block_csv)

    


        
        



