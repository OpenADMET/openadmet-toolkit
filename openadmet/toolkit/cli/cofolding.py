import click
import pandas as pd

from openadmet.toolkit.cofolding.boltz import BoltzCoFoldingEngine
from openadmet.toolkit.cofolding.chai1 import Chai1CoFoldingEngine

@click.command()
@click.option(
    "--model",
    type=click.Choice(["chai1", "boltz"], case_sensitive=False),
    default="boltz",
    help="Model to use for cofolding, must be one of boltz or chai1",
    required=False
)
@click.option(
    "--input_csv",
    type=click.Path(exists=True),
    default=None,
    help="Input CSV file containing fasta files and protein names for cofolding",
    required=False,
)
@click.option(
    "--fasta_column",
    type=str,
    default="fasta",
    help="Column name in the input CSV file containing fasta files for cofolding",
    required=False,
)
@click.option(
    "--protein_name_column",
    type=str,
    default="protein_name",
    help="Column name in the input CSV file containing protein names for cofolding",
    required=False,
)
@click.option(
    "--fastas",
    default=None,
    multiple=True,
    help="Fasta file or list of fasta files for cofolding",
    required=False,
)
@click.option(
    "--protein_names",
    default=None,
    multiple=True,
    help="Protein names for cofolding, by default None",
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
    default=True,
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
@click.option(
    "--num_trunk_recycles",
    type=click.INT,
    default=3,
    help="Number of recycling steps for trunk model",
    required=False,
)
@click.option(
    "--num_diffn_timesteps",
    type=click.INT,
    default=200,
    help="Number of diffusion timesteps for trunk model",
    required=False,
)
@click.option(
    "--seed",
    type=click.INT,
    default=42,
    help="Random seed for reproducibility",
    required=False,
)
@click.option(
    "--use_esm_embeddings",
    type=click.BOOL,
    default=True,
    help="Use ESM embeddings for trunk model",
    required=False,
)
def cofolding(
    model: str,
    input_csv: str,
    fasta_column: str,
    protein_name_column: str,
    fastas: list[str],
    protein_names: list[str],
    output_dir: str,
    device: str,
    use_msa_server: bool,
    diffusion_samples: int,
    recycling_steps: int,
    sampling_steps: int,
    num_trunk_recycles: int,
    num_diffn_timesteps: int,
    seed: int,
    use_esm_embeddings: bool,
):
    """
    Run cofolding on the given fasta files and return the paths to the generated cif files
    """

    # Check if input_csv and fastas are provided
    if input_csv is None and fastas is None:
        raise ValueError("Either input_csv or input_yaml or fastas must be provided")
    
    # Check that only one of input_csv, input_yaml, or fastas is provided
    if input_csv is not None and fastas is not None:
        raise ValueError("Provide only one of input_csv, input_yaml, or fastas.")

    # Check if input_csv is provided and read the fasta files and protein names from it
    if input_csv is not None:
        df = pd.read_csv(input_csv)
        fastas = df[fasta_column].tolist()
        protein_names = df[protein_name_column].tolist()

    # Check if protein_names and fastas are of the same length
    if protein_names is not None and len(fastas) != len(protein_names):
        raise ValueError("Length of fasta and protein_name should be the same")

    # Initialize the cofolding engine
    if model == "chai1":
        cofolding_engine = Chai1CoFoldingEngine(
            output_dir=output_dir,
            device=device,
            use_msa_server=use_msa_server,
            num_trunk_recycles=num_trunk_recycles,
            num_diffn_timesteps=num_diffn_timesteps,
            use_esm_embeddings=use_esm_embeddings,
            seed=seed,
        )

    elif model == "boltz":
        cofolding_engine = BoltzCoFoldingEngine(
            output_dir=output_dir,
            device=device,
            use_msa_server=use_msa_server,
            diffusion_samples=diffusion_samples,
            recycling_steps=recycling_steps,
            sampling_steps=sampling_steps,
        )
    else:
        raise ValueError("Invalid model name. Choose either 'chai1' or 'boltz'.")

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
