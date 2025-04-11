from pathlib import Path
from pydantic import BaseModel, Field


def combine_seq_smiles_to_fasta(
    seqs: list[str],
    names: list[str],
    protein_or_ligand: list[str],
    msa_paths: list[str] = None,
    method: str="chai1",
    unit_stride: int = 2,

) -> str:
    """
    Takes a list of smiles strings and a fasta strings, a list of protein or ligand
    names, and labels of "protein" or "ligand" and combines them into a single string
    that chai1 model can use, looking like:

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
        Either "protein" or "ligand" or "smiles" or "ccd"
    msa_paths: list[str]
        Paths to the MSA files, only for boltz1
    method: str
        The method to use for cofolding, either "chai1" or "boltz1"
    unit_stride: int
        Break up the sequence into chunks of this size

    Returns
    -------
    str
        Combined fasta string
    """

    if not (len(seqs) == len(names) == len(protein_or_ligand)):
        raise ValueError(
            "seqs, names, and protein_or_ligand must all be the same length"
        )
    if msa_paths is not None and len(msa_paths) != len(seqs):
        raise ValueError("msa_paths must be the same length as seqs")

    if not all(item in ["protein", "ligand", "smiles", "ccd"] for item in protein_or_ligand):
        raise ValueError("unexpected tag, must be one of 'protein', 'ligand', 'smiles', 'ccd'")

    if method not in ["chai1", "boltz1"]:
        raise ValueError("unexpected method, must be one of 'chai1', 'boltz1'")

    if method == "chai1":
        fasta_chunks = []
        for seq, name, pl in zip(seqs, names, protein_or_ligand):
            fasta_chunks.append(f">{pl}|name={name}\n{seq}\n")
    elif method == "boltz1":
        fasta_chunks = []
        if msa_paths is not None:
            for seq, name, pl, path in zip(seqs, names, protein_or_ligand, msa_paths):
                fasta_chunks.append(f">{name}|{pl}|{path}\n{seq}\n")
        else:
            for seq, name, pl in zip(seqs, names, protein_or_ligand):
                fasta_chunks.append(f">{name}|{pl}\n{seq}\n")

    # join every unit_stride fasta_chunks
    segments = []
    for i in range(0, len(fasta_chunks), unit_stride):
        seg = "".join(fasta_chunks[i : i + unit_stride])  # noqa: E203
        segments.append(seg)

    return segments


class CoFoldingEngine(BaseModel):
    output_dir: Path = Field(..., description="Output directory to save the results")
    device: str = Field(
        "cuda:0", description="Device to run the model on, torch.device string"
    )
