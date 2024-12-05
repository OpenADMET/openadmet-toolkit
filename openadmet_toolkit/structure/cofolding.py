from pydantic import BaseModel, Field
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from chai_lab.chai1 import run_inference
import os
import tempfile
from shutil import copyfile







class CoFoldingEngine(BaseModel):
    output_dir: Path = Field(..., description="Output directory to save the results")
    device : str = Field("cuda:0", description="Device to run the model on, torch.device string")


class Chai1CoFoldingEngine(CoFoldingEngine):

    

    def inference(self, fasta: str, protein_name: Optional[str]=None):
        # write fasta to tempfile
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
        scores = [rd.aggregate_score for rd in candidates.ranking_data]
    

        # make output_dir if not exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # copy to output_path
        new_cif_paths = []
        for cif_path in cif_paths:
            newpath = self.output_dir / cif_path.name
            copyfile(cif_path, newpath)
            new_cif_paths.append(newpath)


        return new_cif_paths, np.asarray(scores).ravel()



    