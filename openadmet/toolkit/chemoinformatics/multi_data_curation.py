import yaml
import fsspec
from typing import Any, ClassVar, Set
from functools import reduce
import pandas as pd
import os
from pathlib import Path

from openadmet.toolkit.chemoinformatics.data_curation import DataProcessing

class MultiDataProcessing(DataProcessing):
    """
    Class for loading a yaml file which contains all the data files and relevant arguments for multitask processing.
    """
    REQUIRED_KEYS: ClassVar[Set[str]] = {"resource", "smiles_col", "target_col", "activity_type"}

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
            missing_keys = cls.REQUIRED_KEYS - spec.keys()
            if missing_keys:
                raise ValueError(f"Target '{target_name}' is missing required keys: {missing_keys}")

            # Validate non-empty values for all keys except "activity_type"
            for key in cls.REQUIRED_KEYS - {"activity_type"}:
                if spec.get(key) in (None, ""):
                    raise ValueError(f"Target '{target_name}' has empty value for key: '{key}'")

            # Add the validated spec as a dictionary with target_name included
            spec["target_name"] = target_name
            data_entries.append(spec)

        return data_entries
    
    @classmethod
    def batch_process(cls, path:str, pchembl:bool, savefile:bool = False, outputdir:str = ''):
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
        pchembl : bool
            Whether or not the files are from ChEMBL. True if files are from ChEMBL, False if not.
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

            if pchembl:
                df["OPENADMET_LOGAC50"] = df[spec["target_col"]]
                df["OPENADMET_ACTIVITY_TYPE"] = df.apply(lambda row: f"pChEMBL: p{row[spec['activity_type']]}", axis=1)
            else:
                # Get the optional value of input_unit if LOGAC50 needs to be calculated
                input_unit = spec.get("input_unit", "M")
                df = processor.get_pac50(df, 
                                        pac50_col=spec["target_col"], 
                                        input_unit=input_unit, 
                                        activity_type=spec["activity_type"])

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
    def multitask_process(cls, path:str, process:bool, pchembl:bool, savemultifile:bool, multioutputdir:str = ''):
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
        pchembl : bool
            Whether or not the files are from ChEMBL. True if files are from ChEMBL, False if not.
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
            data = cls.batch_process(path=path, pchembl=pchembl)

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

