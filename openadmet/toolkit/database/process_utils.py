from pathlib import Path
from glob import glob 
import requests
import biotite.structure.io as bsio
import numpy as np
import openadmet.toolkit.database.rcsb_utils as rut

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
    "MOO",
    "MPD",
    "IPA",
    "IMD",
    "EDO",
    "DMS",
    "GOL",
    "CIT",
    "PG0",
    "2CV",
    "CPS",
    "ADP",
]

def get_rcsb_data_entry(pdb_id):
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    r = requests.get(url)
    if r.status_code != 200:
        print(f"Failed to receive data entry for {pdb_id}")
        return []
    return r.json()

def get_ligand_chain_ids(entry, pdb_id):
    ligand_ids = entry.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids", [])
    ligand_chains = {}
    chem_ids = []
    for lig_id in ligand_ids:
        url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/{lig_id}"
        r2 = requests.get(url)
        if r2.status_code != 200:
            continue
        data = r2.json()
        # print(data)
        chem_id = data['pdbx_entity_nonpoly']["comp_id"]
        chains = data["rcsb_nonpolymer_entity_container_identifiers"]["auth_asym_ids"]
        ligand_chains[chem_id] = chains   
        chem_ids.append(chem_id)
    return ligand_chains, chem_ids

def get_chain_ids(entry, pdb_id, uniprot):
    entities = entry.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])
    chains = []
    for ent in entities:
        r2 = requests.get(
            f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{ent}"
        )
        if r2.status_code != 200:
            continue
        poly = r2.json()
        for ref in poly.get("rcsb_polymer_entity_container_identifiers", {}).get("reference_sequence_identifiers", []):
            if ref.get("database_accession") == uniprot:
                chains.extend(
                    poly.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
                )
    return sorted(set(chains))

def process_rcsb_pdb(target, input_path="", fmt="cif", outdir="."):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for f in glob(f"{input_path}/*.{fmt}"):
        _id = f.split("/")[-1].split(".")[0]
        print(f"\nProcessing: {_id}")
        final_path = f"{outdir}/{_id}_cleaned.{fmt}"
        entry = get_rcsb_data_entry(_id)
        chain_ids = get_chain_ids(entry, _id, rut.UNIPROT_IDS[target])
        print(f"Checking Chain IDs...")
        if 'A' not in chain_ids:
            print(f"Structure {_id} is selecting other than Chain A.")
        lig_chain_ids, lig_ids = get_ligand_chain_ids(entry, _id)
        lig_ids = [x for x in lig_ids if x not in COMMON_CO_CRYSTALS]
        structure = bsio.load_structure(f)
        prot_chain = chain_ids[0]
        protein_mask = (structure.chain_id == prot_chain) & (structure.hetero == False)
        lig_mask = (structure.chain_id == prot_chain) & np.isin(structure.res_name.astype(str), lig_ids)
        final_mask = protein_mask | lig_mask
        final_structure = structure[final_mask]
        try:
            bsio.save_structure(final_path, final_structure)
            print(f"Saved processed file: {final_path}")
        except Exception as e:
            print(f"Failed to process {_id}: {e}")
