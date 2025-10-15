from glob import glob
from io import StringIO
from pathlib import Path

import gemmi
import MDAnalysis as mda
import requests
from MDAnalysis.lib.util import NamedStream

COMMON_CO_CRYSTALS = [
    "ZN",
    "NA",
    "MG",
    "K",
    "ACT",
    "ER3",
    "SO4",
    "PO4",
    "NI",
    "MOO",  # ions
    "MPD",
    "IPA",
    "IMD",
    "EDO",
    "DMS",
    "GOL",
    "CIT",
    "PG0"  # solvents
    "2CV",
    "CPS",  # detergents
    "ADP",
]  # biological, in AHR


def get_pdb_ids(uniprot_id, rows=1000):
    """
    Fetch PDB IDs associated with a given UniProt ID using the RCSB PDB API.

    Parameters:
    -----------
        uniprot_id (str): The UniProt ID to search for.
        rows (int): Number of rows to return. Default is 1000.

    Returns:
    --------
        list: A list of PDB IDs associated with the UniProt ID.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json="
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id,
            },
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": rows,  # default 10, change if 1000 ids found
            }
        },
    }
    response = requests.post(url, json=query)
    response.raise_for_status()
    result = response.json()
    pdb_ids = [entry["identifier"] for entry in result["result_set"]]
    if len(pdb_ids) < rows:
        print(f"Found {len(pdb_ids)} PDB IDs.")
    else:
        print(
            f"Found {len(pdb_ids)} PDB IDs. Consider changing rows to greater than {rows}."
        )
    return pdb_ids


def get_rcsb_url(pdb_id, fmt="pdb"):
    """
    Fetch the PDB or mmCIF file from RCSB for the given PDB ID.
    """
    url = f"https://files.rcsb.org/download/{pdb_id}.{fmt}"
    return requests.get(url)


def write_file(text, file_path):
    """Write text content to a file."""
    with open(file_path, "w") as f:
        f.write(text)


def convert_cif_to_pdb_gemmi(cif_text):
    """Convert mmCIF text to PDB string using gemmi."""
    doc = gemmi.cif.read_string(cif_text)
    structure = gemmi.make_structure_from_block(doc.sole_block())
    return structure.make_pdb_string()


def get_rcsb_data_entry(pdb_id):
    """Fetch the data entry for a given PDB ID from RCSB."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to receive data entry for {pdb_id}")
        return []
    return r.json()


def get_ligand_chain_ids(entry, pdb_id):
    """Fetch ligand chain IDs from the RCSB data entry.
    Returns a dictionary mapping ligand chemical IDs to their chain IDs.
    """
    ligand_ids = entry.get("rcsb_entry_container_identifiers", {}).get(
        "non_polymer_entity_ids", []
    )
    ligand_chains = {}
    for lig_id in ligand_ids:
        url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{lig_id}"
        r2 = requests.get(url)
        if r2.status_code != 200:
            continue
        data = r2.json()
        # print(data)
        chem_id = data["pdbx_entity_nonpoly"]["comp_id"]
        chains = data["rcsb_nonpolymer_entity_container_identifiers"]["auth_asym_ids"]
        ligand_chains[chem_id] = chains
    return ligand_chains


def get_chain_ids(entry, pdb_id, uniprot):
    """Fetch chain IDs for a given UniProt ID from the RCSB data entry."""
    entities = entry.get("rcsb_entry_container_identifiers", {}).get(
        "polymer_entity_ids", []
    )
    chains = []
    for ent in entities:
        r2 = requests.get(
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{ent}"
        )
        if r2.status_code != 200:
            continue
        poly = r2.json()
        for ref in poly.get("rcsb_polymer_entity_container_identifiers", {}).get(
            "reference_sequence_identifiers", []
        ):
            if ref.get("database_accession") == uniprot:
                chains.extend(
                    poly.get("rcsb_polymer_entity_container_identifiers", {}).get(
                        "auth_asym_ids", []
                    )
                )
    return sorted(set(chains))


def get_rcsb_pdb(pdb_id, outdir=".", download_initial=False):
    """
    Download a PDB or mmCIF for the given PDB ID.
    Returns the final PDB content as a string.
    The initial data file is saved, if specified.
    """
    if download_initial:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        pdb_path = outdir / f"{pdb_id}_initial.pdb"

    # First try direct PDB download
    pdb_response = get_rcsb_url(pdb_id, fmt="pdb")
    if pdb_response.status_code == 200:
        print(f"Downloading {pdb_id} from RCSB...")
        pdb_text = pdb_response.text
        if download_initial:
            print(f"Saving {pdb_id} from RCSB...")
            write_file(pdb_text, pdb_path)
        return pdb_text

    # Fallback to CIF
    print(f"PDB for {pdb_id} not found. Checking for mmCIF...")
    cif_response = get_rcsb_url(pdb_id, fmt="cif")
    if cif_response.status_code == 200:
        print("mmCIF found. Converting to PDB...")
        pdb_text = convert_cif_to_pdb_gemmi(cif_response.text)
        if download_initial:
            write_file(pdb_text, pdb_path)
        return pdb_text

    raise ValueError(f"Neither PDB nor CIF available for {pdb_id}.")


def get_pdb_path(pdb_dir, pdb_id):
    """Get the path to a PDB file in a given directory."""
    return glob(f"{pdb_dir}//*{pdb_id}*.pdb")[0]


def process_pdb(
    pdb_id,
    pdb_text="",
    outdir=".",
    exclude_resnames="",
    chain_ids=None,
    input_path=None,
):
    """
    Process a PDB file to isolate a specific chain and its ligands.
    Ligand chain IDs are converted to "LIG". This formating is to align with MTENN functionality.
    Optionally exclude specified co-crystallized molecules.

    Parameters:
    -----------
        pdb_id (str): The PDB ID of the structure.
        pdb_text (str): The PDB file content as a string. If empty, input_path is used.
        outdir (str): Directory to save the processed PDB file. Default is current directory.
        exclude_resnames (str): Space-separated residue names to exclude (e.g., "SO4 CL").
        chain_ids (list): List of chain IDs to isolate. If None, the first chain is used.
        input_path (str): Path to a directory containing PDB files. If provided, pdb_text is ignored.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    final_path = outdir / f"{pdb_id}.pdb"

    if input_path:
        pdb = get_pdb_path(input_path, pdb_id)
        print(pdb)
        u = mda.Universe(pdb)
    else:
        u = mda.Universe(NamedStream(StringIO(pdb_text), f"{pdb_id}.pdb"))
    protein_A = u.select_atoms(f"protein and chainid {chain_ids[0]}")
    others = u.select_atoms(
        f"chainid {chain_ids[0]} and not protein and not water {exclude_co_crystals}"
    )

    combined = protein_A + others
    lig = combined.select_atoms(
        f"chainid {chain_ids[0]} and not protein and not resname HEM and not resname HEC"
    )

    if len(lig) > 0:
        lig.residues.resnames = ["LIG"] * len(lig.residues)

    final_lig_set = set(
        combined.select_atoms(
            f"chainid {chain_ids[0]} and not protein and not water"
        ).resnames
    )

    combined.write(final_path)
    return final_path
