import requests
from pathlib import Path
import gemmi

UNIPROT_IDS = {
               "CYP2D6": "P10635", 
               "CYP3A4": "P08684",
               "CYP1A2": "P05177",
               "CYP2C9": "P11712", 
               "PXR"   : "O75469", 
               "AHR"   : "P35869"
              }


def get_pdb_ids(uniprot_id, rows=1000):
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json="
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot_id
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": rows  # default 10, change if 1000 ids found 
            }
        }
    }
    response = requests.post(url, json=query)
    response.raise_for_status()
    result = response.json()
    pdb_ids = [entry["identifier"] for entry in result["result_set"]]
    if len(pdb_ids) < rows:
        print(f"Found {len(pdb_ids)} PDB IDs.")
    else:
        print(f"Found {len(pdb_ids)} PDB IDs. Consider changing rows to greater than {rows}.")
    return pdb_ids

def get_rcsb_url(pdb_id, fmt="cif"):
    url = f"https://files.rcsb.org/download/{pdb_id}.{fmt}"
    return requests.get(url)

def download_rcsb_file(pdb_id, fmt='cif', outdir=".", get_text=False):
    """
    Download specified file type (default mmCIF, but works for PDB)
    for the given PDB ID in the specified output directory.  
    Returns file text as string. Set get_text to True if you do not want to download the file, only get the text. 
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    file_path = outdir / f"{pdb_id}.{fmt}"

    url_response = get_rcsb_url(pdb_id, fmt=fmt)
    if url_response.status_code == 200:
        print(f"Downloading {pdb_id} from RCSB...")
        print(f"Saving {pdb_id} from RCSB...")
        text = url_response.text
        if not get_text:
            with open(file_path, "w") as f:
                f.write(text)
        return text
    raise ValueError(f"File type {fmt} not available for {pdb_id}.")

def convert_cif_to_pdb_gemmi(cif_text):
    """Convert mmCIF text to PDB string using gemmi."""
    doc = gemmi.cif.read_string(cif_text)
    structure = gemmi.make_structure_from_block(doc.sole_block())
    return structure.make_pdb_string()
