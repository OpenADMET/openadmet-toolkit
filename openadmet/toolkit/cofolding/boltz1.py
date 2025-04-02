import gc
import os
import tempfile
from pathlib import Path
from shutil import copyfile
from typing import Optional, Union
import subprocess

import numpy as np
import torch

from pydantic import Field

from openadmet.toolkit.cofolding.cofold_base import CoFoldingEngine

class Boltz1CoFoldingEngine(CoFoldingEngine):

    use_msa_server: bool = Field(
                False, description="Use MSA server for multiple sequence alignment"
    )

    def inference(self, fastas):
        """
        Run inference on the given fasta files and return the paths to the generated cif files

        Parameters
        ----------
        fastas : Union[str, list[str]]
            Fasta file or list of fasta files
        protein_names : Optional[Union[str, list[str]]], optional
            Protein names, by default None

        Returns
        -------
        all_paths : list[list[Path]]
            List of list of paths to the generated cif files
        all_scores : np.ndarray
            Array of scores for each protein
        """
        # write fasta to tempfile
        if isinstance(fastas, str):
            fastas = [fastas]

        if protein_names is not None:
            if isinstance(protein_names, str):
                protein_names = [protein_names]
        else:
            protein_names = [f"protein_{i}" for i in range(len(fastas))]

        if len(fastas) != len(protein_names):
            raise ValueError("Length of fasta and protein_name should be the same")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        all_paths = []
        all_scores = []
        for i, (protein_name, fasta) in enumerate(zip(protein_names, fastas)):

            tmpdirname = Path(os.path.join(tempfile.mkdtemp(), "tmp_fold"))
            # make seperate tempdir for fasta
            fasta_path = Path(tempfile.NamedTemporaryFile(suffix=".fasta").name)

            with open(fasta_path, "w") as f:
                f.write(fasta)

            subprocess.run(["boltz", "predict",
                            fasta_path,
                            "--out_dir", tmpdirname,
                            "--cache", f"{tmpdirname}/.boltz",
                            "--checkpoint", "None",
                            "--devices", "1",
                            "--accelerator", "gpu",
                            "--recycling_steps", "3",
                            "--sampling_steps", "200",
                            "--diffusion_samples", "1",
                            "--step_scale", "1.638",
                            "--output_format", "mmcif",
                            "--num_workers", "2",
                            "--override", "False",
                            "--use_msa_server", f"{self.use_msa_server}",
                            "--msa_server_url", "https://api.colabfold.com",
                            "--msa_pairing_strategy", "greedy"])
            
            # clean out gpu_memory
            torch.cuda.empty_cache()

            
            
