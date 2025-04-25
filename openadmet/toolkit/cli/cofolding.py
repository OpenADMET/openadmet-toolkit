import click
import pandas as pd

from openadmet.toolkit.cofolding.boltz1 import Boltz1CoFoldingEngine
from openadmet.toolkit.cofolding.chai1 import Chai1CoFoldingEngine

@click.command()
@click.option(
    "--model",
    type=click.Choice(["chai1", "boltz1"], case_sensitive=False),
    default="boltz1",
    help="Model to use for cofolding, must be one of boltz1 or chai1",
    required=False
)
@click.option(
    "--fastas",
    multiple=True,
    help="Fasta file or list of fasta files",
    required=True,
)
@click.option(
    "--protein_names",
    multiple=True,
    help="Protein names, by default None",
)
@click.option(
    "--output_dir",
    type=click.Path(exists=True),
    default="./",
    help="Output directory for the generated cif files",
    required=True,
)
@click.option(
    "--device",
    type=click.Choice(["cuda:0", "cuda:1", "cpu"], case_sensitive=False),
    default="cuda:0",
    help="Device to run the model on, torch.device string",
    required=False,
)
@click.option(
    "--use_msa_server",
    type=click.BOOL,
    default=False,
    help="Use MSA server for multiple sequence alignment",
    required=False,
)
@click.option(
    "--diffusion_samples",
    type=click.INT,
    default=1,
    help="Number of diffusion samples",
    required=False,
)
@click.option(
    "--recycling_steps",
    type=click.INT,
    default=3,
    help="Number of recycling steps",
    required=False,
)
@click.option(
    "--sampling_steps",
    type=click.INT,
    default=200,
    help="Number of sampling steps",
    required=False,
)
def cofolding(
    model: str,
    fastas: list[str],
    protein_names: list[str],
    output_dir: str,
    device: str,
    use_msa_server: bool,
    diffusion_samples: int,
    recycling_steps: int,
    sampling_steps: int,
):
    """
    Run cofolding on the given fasta files and return the paths to the generated cif files
    """

    # Check if protein_names and fastas are of the same length
    if protein_names is not None and len(fastas) != len(protein_names):
        raise ValueError("Length of fasta and protein_name should be the same")

    # Initialize the cofolding engine
    if model == "chai1":
        cofolding_engine = Chai1CoFoldingEngine(
            output_dir=output_dir,
            device=device,
        )
    elif model == "boltz1":
        cofolding_engine = Boltz1CoFoldingEngine(
            output_dir=output_dir,
            device=device,
            use_msa_server=use_msa_server,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
        )
    else:
        raise ValueError("Invalid model name. Choose either 'chai1' or 'boltz1'.")

    # Inference
    paths, scores = cofolding_engine.inference(
        fastas=fastas,
        protein_names=protein_names,
    )
    print("Paths to the generated cif files:")

    for i, path in enumerate(paths):
        print(f"Protein {i}: {path}")

    score_df = pd.DataFrame(scores, columns=["score"])
    score_df.to_csv(f"{output_dir}/confidence_scores.csv", index=False)
