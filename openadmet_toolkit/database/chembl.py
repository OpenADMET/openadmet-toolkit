import chembl_downloader
from typing import Tuple
from pathlib import Path


def download_extract_chembl_sqlite(version=None) -> Tuple[str, Path]:
    """Download the ChEMBL SQLite database."""
    version, path =  chembl_downloader.download_extract_sqlite(version=version)
    return version, path



