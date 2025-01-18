import gc
import os
import tempfile
from pathlib import Path
from shutil import copyfile
from typing import Optional, Union

import numpy as np
import torch
from chai_lab.chai1 import run_inference
from pydantic import BaseModel, Field


def combine_seq_smiles_to_fasta(
    seqs: list[str],
    names: list[str],
    protein_or_ligand: list[str],
) -> str:
    """
    Takes a list of smiles strings and a fasta strings, a list of protein or ligand names, and labels of "protein" or "ligand", and combines them into a single string that chai1 model can use
    looking like

    >protein|name=example-peptide
    GAAL
    >ligand|name=example-ligand-as-smiles

    Parameters
    ----------
    seqs : list[str]
        Fasta string or Smiles string
    names: list[str]
        Name of protein or ligand
    protein_or_ligand: list[str]
        Either "protein" or "ligand"

    Returns
    -------
    str
        Combined fasta string
    """

    if not len(seqs) == len(names) == len(protein_or_ligand):
        raise ValueError(
            "seqs, names, and protein_or_ligand must all be the same length"
        )

    if not all(protein_or_ligand) in ["protein", "ligand"]:
        raise ValueError("unexpected tag, must be one of 'protein', 'ligand'")

    to_return = ""
    for seq, name, pl in zip(seqs, names, protein_or_ligand):
        to_return += f">{pl}|name={name}\n{seq}\n"
    return to_return


class CoFoldingEngine(BaseModel):
    output_dir: Path = Field(..., description="Output directory to save the results")
    device: str = Field(
        "cuda:0", description="Device to run the model on, torch.device string"
    )


class Chai1CoFoldingEngine(CoFoldingEngine):
    """
    CoFoldingEngine for Chai1 model, see https://github.com/chaidiscovery/chai-lab
    and paper here https://www.biorxiv.org/content/10.1101/2024.10.10.615955v1
    """

    def inference(
        self,
        fastas: Union[str, list[str]],
        protein_names: Optional[Union[str, list[str]]] = None,
    ):
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

            # run inference, output_dir made inside run_inference
            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=tmpdirname,
                # 'default' setup
                num_trunk_recycles=3,
                num_diffn_timesteps=200,
                seed=42,
                device=torch.device(self.device),
                use_esm_embeddings=True,
            )
            # clean out gpu_memory
            torch.cuda.empty_cache()

            cif_paths = candidates.cif_paths
            scores = np.asarray(
                [rd.aggregate_score for rd in candidates.ranking_data]
            ).ravel()
            all_scores.append(scores)

            del candidates

            new_cif_paths = []
            for j, cif_path in enumerate(cif_paths):
                new_cif_path = self.output_dir / f"{protein_name}_{j}.cif"
                copyfile(cif_path, new_cif_path)
                new_cif_paths.append(new_cif_path)

            all_paths.append(new_cif_paths)

            gc.collect()

        return np.asarray(all_paths), np.asarray(all_scores)
