import pandas as pd
from pydantic import BaseModel, Field
from typing import Optional, Any, ClassVar
from rdkit import Chem
from pathlib import Path
import numpy as np
import yaml
import fsspec
from functools import reduce
import os

from openadmet.toolkit.chemoinformatics.rdkit_funcs import canonical_smiles, smiles_to_inchikey
from openadmet.toolkit.chemoinformatics.activity_funcs import calculate_pac50

class DataProcessing(BaseModel):
    """
    Class to handle processing data from any data file in the following formats:
    .csv, .tsv, .parquet, .xls, .xlsx, .json.

    """
    # smiles_col: Optional[str] = None

    @staticmethod
    def read_file(path):
        """
        Wrapper for detecting file extension and reading in with pandas.
        """
        path = Path(path)
        ext = path.suffix.lower()

        if ext == ".csv":
            return pd.read_csv(path, sep=",")
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        elif ext == ".parquet":
            return pd.read_parquet(path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}. \n Must be one of: .csv, .tsv, .parquet, .xls, .xlsx, .json")

    def standardize_smiles_and_convert(self, data, smiles_col:str):
        """
        Converts data to canonical SMILES and determines InChI key

        Parameters
        ----------
        data : dataframe
            Compound activity data read in from a file
        smiles_col : str
            A column with SMILES strings for canonicalization

        Returns
        -------
        dataframe
            Compound with added activity columns, OPENADMET_CANONICAL_SMILES and OPENADMET_INCHIKEY

        Raises
        ------
        ValueError
            An error checking if the column name provided is actually in the provided dataframe.
        """

        if smiles_col in data.columns:
            data["OPENADMET_CANONICAL_SMILES"] = data[smiles_col].apply(canonical_smiles)
            data["OPENADMET_INCHIKEY"] = data["OPENADMET_CANONICAL_SMILES"].apply(smiles_to_inchikey)
            data.dropna(subset="OPENADMET_INCHIKEY", inplace=True)
        else:
            raise ValueError("The provided column is not in the data table!")
        return data

    def standardize_pac50(self, data, transform:bool, ac50_col:str, activity_type:str, input_unit:str=''):
        """A function to create two standard columns: OPENADMET_LOGAC50 and OPENADMET_ACTIVITY_TYPE.
        If transform = True:
            This function will calculate the pAC50 value from an activity measure.
            This value will be used for future modeling prediction.
            Valid activity measures include but are not limited to: IC50, EC50, XC50, AC50, potency, etc.

        Otherwise, if your activity measure is already log transformed, this function will simply rename the activity measure column to standard naming and create an activity type column


        Args:
            data (dataframe): Dataframe containing all small molecule activity data for a target
            transform (bool): Whether or not perform the log transformation
            pac50_col (str): Name of column in dataframe with activity measure
            input_unit (str): Units of your activity measure, must be one of M, mM, uM, µM, nM
            activity_type (str): Designate the type of activity, e.g. IC50, EC50, XC50, AC50, etc.; can be a column in the dataframe OR any other string that is descriptive of the activity type

        Raises:
            ValueError: An error checking if the column name provided is actually in the provided dataframe.

        Returns:
            dataframe: Two new columns added to provided dataframe: OPENADMET_LOGAC50 (pAC50) and OPENADMET_ACTIVITY_TYPE (type of activity measure)
        """
        # Check that column is in dataframe
        if ac50_col in data.columns:
            # If it is, check that the data type is correct
            def safe_pac50(x):
                try:
                    return calculate_pac50(activity=float(x), input_unit=input_unit)
                except (ValueError, TypeError):
                    return np.nan

            if transform:
                data["OPENADMET_LOGAC50"] = data[ac50_col].apply(safe_pac50)
                if activity_type in data.columns:
                    data["OPENADMET_ACTIVITY_TYPE"] = data[activity_type].apply(lambda x: f"p{x}")
                else:
                    data["OPENADMET_ACTIVITY_TYPE"] = f"p{activity_type}"
            else:
                data["OPENADMET_LOGAC50"] = data[ac50_col].astype(float)
                if activity_type in data.columns:
                    data["OPENADMET_ACTIVITY_TYPE"] = data[activity_type].apply(lambda x: f"{x}")
                else:
                    data["OPENADMET_ACTIVITY_TYPE"] = f"{activity_type}"
        else:
            raise ValueError(f"Oospie-daisy! The provided activity column {ac50_col} is not in the dataframe!")
        return data

class MultiDataProcessing(DataProcessing):
    """
    Class for loading a yaml file which contains all the data files and relevant arguments for multitask processing.
    """
    _REQUIRED_KEYS: ClassVar[set[str]] = {"resource", "smiles_col", "target_col", "activity_type"}

    @classmethod
    def load_yaml(cls, path: str, **storage_options) -> list[dict[str, Any]]:
        """Read in a YAML file containing all the relevant meta info for processing multiple data sets into a single for multitask modeling.

        Parameters
        ----------
        path : str
            Path to the yaml file containing the metadata for data files, column arguments, etc.

        Returns
        -------
        list[dict[str, Any]]
            List of dictionaries of key-value pairs from the yaml file.

        Raises
        ------
        ValueError
            YAML file must contain a top-level 'data' key.
        ValueError
            Each protein target MUST have the following keys:
                resource: path to data file, can be .csv, .tsv, .xls, .xlsx, .parquet
                smiles_col: the name of the column containing SMILES strings
                target_col: the name of the column containing the target value, i.e. activity measure (e.g. IC50, EC50, Ki, etc.)
                activity_type: the name of the column containing the type of activity (e.g. IC50, EC50, Ki, etc.) OR manually provide the type of activity as a string
        ValueError
            Value cannot be empty for the above required keys
        """
        path = Path(path)
        # Open file stream
        with fsspec.open(path, "r", **storage_options) as stream:
            # Safe load the model from stream
            content = yaml.safe_load(stream)

        if not isinstance(content, dict) or "data" not in content:
            raise ValueError("YAML file must contain a top-level 'data' key.")

        data_entries = []
        yaml_dir = path.parent.resolve()

        for target_name, spec in content["data"].items():
            # Convert resource path to absolute (resolved relative to YAML location)
            if "resource" in spec:
                resource_path = Path(spec["resource"])
                if not resource_path.is_absolute():
                    spec["resource"] = str((yaml_dir / resource_path).resolve())

            # Ensure all required keys are present
            missing_keys = cls._REQUIRED_KEYS - spec.keys()
            if missing_keys:
                raise ValueError(f"Target '{target_name}' is missing required keys: {missing_keys}")

            # Validate non-empty values for all keys except "activity_type"
            for key in cls._REQUIRED_KEYS - {"activity_type"}:
                if spec.get(key) in (None, ""):
                    raise ValueError(f"Target '{target_name}' has empty value for key: '{key}'")

            # Add the validated spec as a dictionary with target_name included
            spec["target_name"] = target_name
            data_entries.append(spec)

        return data_entries

    @classmethod
    def batch_process(cls, path:str, log_transform:bool, savefile:bool = False, outputdir:str = ''):
        """A function to process multiple compound activity data files from different target proteins for model training.

        Example use: You have a set of files of compound activity data for various protein targets.
        1) Create a "data.yaml" file that refers to various information that is necessary for data processing with the below structure:
            data:
                cyp2j2: # protein target name, must be unique in YAML file if listing multiple data files
                    resource: processed_full/processed_cyp2j2.parquet # path/to/datafile
                    smiles_col: smiles # column name containing SMILES strings for canonicalization
                    target_col: pchembl_value # column name containing target value for modeling prediction, usually the activity measure
                    activity_type: standard_type # EITHER 1) column name containing the specific activity measure or 2) string containing the type of activity measure, one of (EC50, IC50, XC50, AC50)
                    input_unit: M # [OPTIONAL] the units of the activity measure, one of ("M", "mM", "uM", "µM", or "nM"), necessary if you are not using pchembl values and need to calculate OPENAMDET_LOGAC50

        2) To curate the data, an example call is:
            data_dict = MultiDataProcessing.batch_process(path="data.yaml", pchembl=True, savefile=True, outputdir="processed_ic50")

        Parameters
        ----------
        path : str
            Path to the yaml file containing the metadata for data files, column arguments, etc.
        log_transform : bool
            Whether or not the activity measures need log transformation. E.g. if using pchembl values, the activity measure is already log transformed. True if yes, False if not.
        savefile : bool, Optional
            Whether or not to save the processed files, defaults to False.
        outdir : str, Optional
            Directory to save the processed files if savefile is True. If the directory does not already exist, it will be created.

        Returns
        -------
        dict
            Cleaned and processed dataframes for modeling, each exported to parquet.
        """

        keep_cols = [
            "OPENADMET_INCHIKEY",
            "OPENADMET_CANONICAL_SMILES",
            "OPENADMET_LOGAC50",
            "OPENADMET_ACTIVITY_TYPE"
        ]

        # Get the data files and arguments
        data_specs = cls.load_yaml(path)
        # Instantiate the DataProcessing
        processor = cls()

        # Make a dictionary to store the processed dataframes
        data_dict = {}

        for spec in data_specs:
            # Read in the file
            df = processor.read_file(spec["resource"])

            # Make all the relevant columns
            df = processor.standardize_smiles_and_convert(df, smiles_col=spec["smiles_col"])

            input_unit = spec.get("input_unit", "M")
            df = processor.standardize_pac50(df,
                                             ac50_col=spec["target_col"],
                                             transform=log_transform,
                                             activity_type=spec["activity_type"],
                                             input_unit=input_unit
                                             )

            # Average the duplicates
            clean_df = (
                df[keep_cols]
                .groupby("OPENADMET_INCHIKEY", as_index=False)
                .agg({
                    "OPENADMET_CANONICAL_SMILES": "first",
                    "OPENADMET_LOGAC50": "mean",
                    "OPENADMET_ACTIVITY_TYPE": "first"
                })
            )

            data_dict[spec["target_name"]] = clean_df

            if savefile:
                # Check that the directory exists, if not create it
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
                clean_df.to_parquet(os.path.join(outputdir, f"processed_{spec['target_name']}.parquet"), index=False)
        return data_dict

    @classmethod
    def multitask_process(cls, path:str, process:bool, log_transform:bool, savemultifile:bool, multioutputdir:str = ''):
        """A function to process multiple compound activity data files from different target proteins into a SINGLE file for multitask model training.

        This function can curate raw data files if process = True and then concatenate the processed files for multitasking. If your data files are already curated, this function can do just multitask concatenation with process = False.

        Example use: You have a set of files of compound activity data for various protein targets.
        1) Create a "data.yaml" file that refers to various information that is necessary for data processing with the below structure:
            data:
                cyp2j2: # protein target name, must be unique in YAML file if listing multiple data files
                    resource: processed_full/processed_cyp2j2.parquet # path/to/datafile
                    smiles_col: smiles # column name containing SMILES strings for canonicalization
                    target_col: pchembl_value # column name containing target value for modeling prediction, usually the activity measure
                    activity_type: standard_type # EITHER 1) column name containing the specific activity measure or 2) string containing the type of activity measure, one of (EC50, IC50, XC50, AC50)
                    input_unit: M # [OPTIONAL] the units of the activity measure, one of ("M", "mM", "uM", "µM", or "nM"), necessary if you are not using pchembl values and need to calculate OPENAMDET_LOGAC50

        2) To curate the data, an example call is:
        data_dict = MultiDataProcessing.multitask_process(path="data.yaml", pchembl=True, savemultifile=True, multioutputdir="multitask_ic50")

        Parameters
        ----------
        path : str
            Path to the yaml file containing the metadata for data files, column arguments, etc.
        process: bool
            Whether or not to process the data files or simply combine already processed files.
        log_transform : bool
            Whether or not the activity measures need log transformation. E.g. if using pchembl values, the activity measure is already log transformed. True if yes, False if not.
        savemultifile : bool
            Whether or not to save the multitask file.
        multioutputdir : str
            Directory to save the processed files if savefile is True. If the directory does not already exist, it will be created.

        Returns
        -------
        pd.DataFrame
            Merged and cleaned dataframe of compound activity data for multitask modeling
        """

        if process:
            data = cls.batch_process(path=path, log_transform=log_transform)

            # Append the target name to OPENADMET_LOGAC50
            for i in data:
                data[i] = data[i].rename(columns = {"OPENADMET_LOGAC50": f"OPENADMET_LOGAC50_{i}", "OPENADMET_ACTIVITY_TYPE": f"OPENADMET_ACTIVITY_TYPE_{i}"})
        else:
            # Get the data files and arguments
            data_specs = cls.load_yaml(path)
            # Instantiate the DataProcessing
            processor = cls()

            # Make a dict to store the data
            data = {}

            for spec in data_specs:
                # Read in the file
                df = processor.read_file(spec["resource"])

                # Check that df has all the right columns
                keep_cols = [
                    "OPENADMET_INCHIKEY",
                    "OPENADMET_CANONICAL_SMILES",
                    "OPENADMET_LOGAC50",
                    "OPENADMET_ACTIVITY_TYPE"
                ]

                missing_cols = [x for x in keep_cols if x not in df.columns]
                if missing_cols:
                    raise ValueError(f"Error! Missing required columns: {missing_cols}. Reprocess your data.")

                data[spec["target_name"]] = df
                # Append the target name to OPENADMET_LOGAC50
                for i in data:
                    data[i] = data[i].rename(columns = {"OPENADMET_LOGAC50": f"OPENADMET_LOGAC50_{i}", "OPENADMET_ACTIVITY_TYPE": f"OPENADMET_ACTIVITY_TYPE_{i}"})

        data_list = list(data.values())

        merged = reduce(
            lambda left, right: pd.merge(
                left, right, on = ["OPENADMET_INCHIKEY", "OPENADMET_CANONICAL_SMILES"],
                how = "outer"
            ),
            data_list
        )

        if savemultifile:
            if not os.path.exists(multioutputdir):
                os.makedirs(multioutputdir)
            merged.to_parquet(os.path.join(multioutputdir, "multitask.parquet"), index=False)

        return merged

class CSVProcessing(BaseModel):
    """
    Class to handle processing data from a csv downloaded

    """
    smiles_col: Optional[str] = None

    @staticmethod
    def read_csv(csv_path, sep=","):
        """
        Wrapper for inbuilt pandas read_csv()
        """
        return pd.read_csv(csv_path, sep=sep)

    def standardize_smiles_and_convert(self, data):
        """
        Converts data to canonical smiles and determines inchikey

        Parameters
        ----------
        data : DataFrame
            Dataframe of csv of downloaded compound data

        Returns
        -------
        data : DataFrame
            Dataframe with smiles canonicalized and inchikey
            column added
        """

        if self.smiles_col:
            if self.smiles_col not in data.columns:
                raise ValueError("The provided column is not in the data table!")
            else:
                col = self.smiles_col

        else:
            # Get column with valid SMILES string
            cols = []
            for i in data.columns:
                for val in data[i]:
                    if pd.notna(val):
                        mol = Chem.MolFromSmiles(str(val))
                        if mol is not None:
                            cols.append(i)
                        break

            if len(cols) == 1:
                col = cols[0]
            else:
                raise ValueError(f"Multiple columns with SMILES strings detected! Choose one for OPENADMET_CANONICAL_SMILES: {cols}.")

        data["OPENADMET_CANONICAL_SMILES"] = data[col].apply(lambda x: canonical_smiles(x))
        data["OPENADMET_INCHIKEY"] = data["OPENADMET_CANONICAL_SMILES"].apply(
            lambda x: smiles_to_inchikey(x)
        )
        data.dropna(subset="OPENADMET_INCHIKEY", inplace=True)
        return data


class ChEMBLProcessing(CSVProcessing):
    """
    Class to handle processing data from a csv downloaded
    from ChEMBL

    Fields
    ------
    inhib : bool
        Indicates if data is for inhibition
    react : bool
        Indicates if data is for reactivity/substrate
    min_compound_num : int
        Minimum number of compounds threshold (determined by molecule_count)
    pchembl_thresh : float
        Threshold pchembl value
    min_assay_num : int
        Minimum number of assays threshold (determined by assay_count)
    save_as : str
        Path/filename to save processed data csv
    keep_cols_inhib : list[str]
        List of ChEMBL columns to keep in processed data (default below)

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
            "OPENADMET_CANONICAL_SMILES",
            "OPENADMET_INCHIKEY",
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
            "OPENADMET_CANONICAL_SMILES",
            "OPENADMET_INCHIKEY",
            "Molecule Name",
            "Action Type",
        ]
    )

    def process(self, path):
        """
        Process raw ChEMBL data to clean up row names, canonicalize smiles,
        add inchikeys, threshold for better compounds, dedupilcate, and
        save new data

        Parameters
        ----------
        path : str
            Path to ChEMBL csv to process

        Returns
        -------
        df : DataFrame
            Processed ChEMBL dataframe
        """
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

        more_than_N_compounds.OPENADMET_INCHIKEY = more_than_N_compounds.OPENADMET_INCHIKEY.astype(str)
        num_assays_per_compound_df.OPENADMET_INCHIKEY = (
            num_assays_per_compound_df.OPENADMET_INCHIKEY.astype(str)
        )

        combined_df = more_than_N_compounds.merge(
            num_assays_per_compound_df, on="OPENADMET_INCHIKEY"
        )
        combined_df.sort_values("assay_count", ascending=False, inplace=True)
        combined_df["assay_count"] = combined_df["assay_count"].astype(int)

        compound_grouped_mean = combined_df.groupby("OPENADMET_INCHIKEY")["pChEMBL Value"].mean()
        compound_grouped_mean.reset_index()

        cgm = compound_grouped_mean.reset_index(name="pChEMBL mean")
        cgm = cgm.set_index("OPENADMET_INCHIKEY")
        combined_df = combined_df.join(cgm, on="OPENADMET_INCHIKEY")

        compound_grouped_std = combined_df.groupby("OPENADMET_INCHIKEY")["pChEMBL Value"].std()

        cgstd = compound_grouped_std.reset_index(name="pChEMBL std")
        cgstd = cgstd.set_index("OPENADMET_INCHIKEY")
        combined_df = combined_df.join(cgstd, on="OPENADMET_INCHIKEY")

        # get active compounds
        # defined as compounds above pChEMBL value specified (default 5.0)
        if pchembl_thresh is not None:
            active = combined_df[combined_df["pChEMBL mean"] >= pchembl_thresh]
        else:
            active = combined_df.copy()
        clean_deduped = self.clean_and_dedupe_actives(active, save_as)
        if min_assay_num is not None:
            return self.more_than_L_assays(clean_deduped, min_assay_num)
        else:
            return clean_deduped

    def select_quality_data_reactivity(self, data, save_as):
        substrates = data[data["Action Type"] == "SUBSTRATE"]
        substrates = self.clean_and_dedupe_actives(substrates, save_as)
        return substrates

    def more_than_L_assays(self, clean_deduped, min_assay_num, save_as=None):
        if min_assay_num is not None:
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
            subset="OPENADMET_INCHIKEY", keep="first"
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
        if min_compound_num is not None:
            more_than_N_compounds = combined[
                combined["molecule_count"] > min_compound_num
            ]
            return more_than_N_compounds
        else:
            return combined

    def get_num_assays_per_compound(self, more_than_N_compounds):
        num_assays_per_compound_df = (
            more_than_N_compounds.groupby(["OPENADMET_INCHIKEY"])["Assay ChEMBL ID"]
            .size()
            .reset_index(name="assay_count")
        )
        num_assays_per_compound_df.set_index("OPENADMET_INCHIKEY")
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
        compound_grouped_mean = combined_df.groupby("OPENADMET_INCHIKEY")["pChEMBL Value"].mean()
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
            "OPENADMET_CANONICAL_SMILES",
            "OPENADMET_INCHIKEY",
            "PUBCHEM_ACTIVITY_OUTCOME",
            "PUBCHEM_CID",
        ]
    )

    def process(self, path, aid, data_type, save_as=None):
        """
        Process raw PubChem data to clean up row names, canonicalize smiles,
        add inchikeys, dedupilcate, and save new data

        Parameters
        ----------
        path : str
            Path to PubChem csv to process

        Returns
        -------
        df : DataFrame
            Processed PubChem dataframe
        """
        data = self.read_csv(path)
        data.rename(columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "Smiles"}, inplace=True)
        data = self.delete_metadata_rows(data)
        data = data.dropna(subset="PUBCHEM_CID")
        data["PUBCHEM_SID"] = data["PUBCHEM_SID"].astype(int)
        data["PUBCHEM_CID"] = data["PUBCHEM_CID"].astype(int)
        data = self.standardize_smiles_and_convert(data)
        data.dropna(subset="OPENADMET_INCHIKEY")
        data = data[self.keep_cols]
        data["dataset"] = aid
        data["data_type"] = data_type
        data["active"] = data["PUBCHEM_ACTIVITY_OUTCOME"] == "Active"
        data = data.drop_duplicates(subset="OPENADMET_INCHIKEY")
        if self.inhib:
            data["action_type"] = "inhibition"
        elif self.react:
            data["action_type"] = "substrate"
        else:
            raise ValueError("Must specify either inhib or react as True.")
        if save_as is not None:
            data.to_csv(save_as, index=False)
        return data

    @staticmethod
    def delete_metadata_rows(data):
        """
        Deletes metadata rows from PubChem csv

        Parameters
        ----------
        data : DataFrame
            Pubchem dataframe read from csv

        Returns
        -------
        data : DataFrame
            DataFrame with non-data rows deleted
        """
        to_del = 0
        for index, row in data.iterrows():
            if index == 0:
                continue
            elif Chem.MolFromSmiles(str(row["Smiles"])) is None:
                to_del += 1
            else:
                break
        data = data.drop(labels=list(range(0, to_del)), axis=0).reset_index(drop=True)
        return data
