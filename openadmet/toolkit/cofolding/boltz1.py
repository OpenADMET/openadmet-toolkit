import gc
import os
import tempfile
import pandas as pd
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
    diffusion_samples: int = Field(
                1, description="Number of diffusion samples"
    )
    recycling_steps: int = Field(
                3, description="Number of recycling steps"
    )
    sampling_steps: int = Field(
                200, description="Number of sampling steps"
    )

    def inference(self,
        fastas: Union[str, list[str]],
        protein_names: Optional[Union[str, list[str]]] = None,):
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

            tmpdirname = Path(tempfile.mkdtemp(
                prefix=f".boltz1_{protein_name}", dir=self.output_dir
            ))
            tmpdirname.mkdir(parents=True, exist_ok=True)
            # make seperate tempdir for fasta
            fasta_path = tmpdirname / f"input_{protein_name}.fasta"

            with open(fasta_path, "w") as f:
                f.write(fasta)

            args = ["boltz", "predict",
                            fasta_path,
                            "--out_dir", tmpdirname,
                            # "--cache", f"{self.output_dir /".boltz"}", ch
                            # "--checkpoint", "None", we can specify a checkpoint later if we do some fine-tuning
                            "--devices", "1",
                            "--accelerator", "gpu",
                            "--recycling_steps", str(self.recycling_steps),
                            "--sampling_steps", str(self.sampling_steps),
                            "--diffusion_samples", str(self.diffusion_samples),
                            "--step_scale", "1.638",
                            "--output_format", "mmcif",
                            "--num_workers", "2",
                            "--override",
                            "--msa_server_url", "https://api.colabfold.com",
                            "--msa_pairing_strategy", "greedy"]

            if self.use_msa_server:
                args.append("--use_msa_server")

            subprocess.run(args)

            # clean out gpu_memory
            torch.cuda.empty_cache()

            fasta_name  = fasta_path.stem

            temp_out_path = tmpdirname / f"boltz_results_{fasta_name}/predictions/{fasta_name}"

            cif_paths = []
            scores = []
            for i in range(self.diffusion_samples):
                cif_path = f"{temp_out_path}/{fasta_name}_model_{i}.cif"
                score = pd.read_json(f"{temp_out_path}/confidence_{fasta_name}_model_{i}.json")["confidence_score"].values
                cif_paths.append(cif_path)
                scores.append(score[0])


            new_cif_paths = []
            for i, cif_path in enumerate(cif_paths):
                new_cif_path = self.output_dir / f"{protein_name}_{i}.cif"
                copyfile(cif_path, new_cif_path)
                new_cif_paths.append(new_cif_path)

            all_paths.append(new_cif_paths)
            all_scores.append(scores)

            gc.collect()

        return np.asarray(all_paths), np.asarray(all_scores)
