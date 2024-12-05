from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from chai_lab.chai1 import run_inference
import os
import tempfile
from shutil import copyfile




def combine_seq_smiles(seq_string: str, smiles: str):
    """
    Takes a smiles string and a fasta string and combines them into a single string that chai1 model can use
    looking like 

    >protein|name=example-peptide
    GAAL
    >ligand|name=example-ligand-as-smiles

    Parameters
    ----------
    fasta : str
        Fasta string
    smiles : str
        Smiles string
    """
    return f">protein|name=protein\n{seq_string}\n>ligand|name=ligand\n{smiles}"


class CoFoldingEngine(BaseModel):
    output_dir: Path = Field(..., description="Output directory to save the results")
    device : str = Field("cuda:0", description="Device to run the model on, torch.device string")


class Chai1CoFoldingEngine(CoFoldingEngine):
    """
    CoFoldingEngine for Chai1 model
    """

    

    def inference(self, fastas: Union[str, list[str]], protein_names: Optional[Union[str, list[str]]]=None):
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
        for i, (protein_name,fasta) in enumerate(zip(protein_names, fastas)):

            tmpdirname = Path(os.path.join(tempfile.mkdtemp(), 'tmp_fold'))
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
            cif_paths = candidates.cif_paths
            scores = np.asarray([rd.aggregate_score for rd in candidates.ranking_data]).ravel()
            all_scores.append(scores)
        
            new_cif_paths = []
            for j, cif_path in enumerate(cif_paths):
                new_cif_path = self.output_dir / f"{protein_name}_{j}.cif"
                copyfile(cif_path, new_cif_path)
                new_cif_paths.append(new_cif_path)

            all_paths.append(new_cif_paths)

        return np.asarray(all_paths), np.asarray(all_scores)










    