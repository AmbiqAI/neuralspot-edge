import gzip
import os
import pickle
from typing import Any

import requests
from tqdm import tqdm

def download_file(src: str, dst: os.PathLike, progress: bool = True, chunk_size: int = 8192):
    """Download file from supplied url to destination streaming.

    Args:
        src (str): Source URL path
        dst (PathLike): Destination file path
        progress (bool, optional): Display progress bar. Defaults to True.

    """
    with requests.get(src, stream=True, timeout=3600 * 24) as r:
        r.raise_for_status()
        req_len = int(r.headers.get("Content-length", 0))
        prog_bar = tqdm(total=req_len, unit="iB", unit_scale=True) if progress else None
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                if prog_bar:
                    prog_bar.update(len(chunk))
            # END FOR
        # END WITH
    # END WITH



def load_pkl(file: str, compress: bool = True) -> dict[str, Any]:
    """Load pickled file.

    Args:
        file (str): File path (.pkl)
        compress (bool, optional): If file is compressed. Defaults to True.

    Returns:
        dict[str, Any]: Dictionary of pickled objects
    """
    if compress:
        with gzip.open(file, "rb") as fh:
            return pickle.load(fh)
    else:
        with open(file, "rb") as fh:
            return pickle.load(fh)


def save_pkl(file: str, compress: bool = True, **kwargs):
    """Save python objects into pickle file.

    Args:
        file (str): File path (.pkl)
        compress (bool, optional): Whether to compress file. Defaults to True.
    """
    if compress:
        with gzip.open(file, "wb") as fh:
            pickle.dump(kwargs, fh, protocol=4)
    else:
        with open(file, "wb") as fh:
            pickle.dump(kwargs, fh, protocol=4)
