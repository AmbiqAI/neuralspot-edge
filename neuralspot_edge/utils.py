import gzip
import logging
import os
import pickle
import random
from typing import Any

import numpy as np
import numpy.typing as npt
import requests
from rich.logging import RichHandler
from tqdm import tqdm


def set_random_seed(seed: int | None = None) -> int:
    """Set random seed across libraries: TF, Numpy, Python

    Args:
        seed (int | None, optional): Random seed state to use. Defaults to None.

    Returns:
        int: Random seed
    """
    seed = seed or np.random.randint(2**16)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import keras  # pylint: disable=import-outside-toplevel
    except ImportError:
        pass
    else:
        keras.utils.set_random_seed(seed)
    return seed


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


def setup_logger(log_name: str) -> logging.Logger:
    """Setup logger with Rich

    Args:
        log_name (str): Logger name

    Returns:
        logging.Logger: Logger
    """
    logger = logging.getLogger(log_name)
    if logger.handlers:
        return logger
    logging.basicConfig(level=logging.ERROR, force=True, handlers=[RichHandler()])
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.handlers = [RichHandler()]
    return logger


def env_flag(env_var: str, default: bool = False) -> bool:
    """Return the specified environment variable coerced to a bool, as follows:
    - When the variable is unset, or set to the empty string, return `default`.
    - When the variable is set to a truthy value, returns `True`.
      These are the truthy values:
          - 1
          - true, yes, on
    - When the variable is set to the anything else, returns False.
       Example falsy values:
          - 0
          - no
    - Ignore case and leading/trailing whitespace.

    Args:
        env_var (str): Environment variable name
        default (bool, optional): Default value. Defaults to False.

    Returns:
        bool: Value of environment variable
    """
    environ_string = os.environ.get(env_var, "").strip().lower()
    if not environ_string:
        return default
    return environ_string in ["1", "true", "yes", "on"]


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

SpecType = list[npt.ArrayLike] | tuple[npt.ArrayLike] | dict[str, npt.ArrayLike] | npt.ArrayLike


def matches_spec(o: SpecType, spec: SpecType, ignore_batch_dim: bool = False) -> bool:
    """Test whether data object matches the desired spec.

    Args:
        o (SpecType): Data object.
        spec (SpecType): Metadata for describing the the data object.
        ignore_batch_dim: Ignore first dimension when checking shapes.

    Returns:
        bool: If matches
    """
    if isinstance(spec, (list, tuple)):
        if not isinstance(o, (list, tuple)):
            raise ValueError(f"data object is not a list or tuple which is required by the spec: {spec}")
        if len(spec) != len(o):
            raise ValueError(f"data object has a different number of elements than the spec: {spec}")
        for i, ispec in enumerate(spec):
            if not matches_spec(o[i], ispec, ignore_batch_dim=ignore_batch_dim):
                return False
        return True

    if isinstance(spec, dict):
        if not isinstance(o, dict):
            raise ValueError(f"data object is not a dict which is required by the spec: {spec}")
        if spec.keys() != o.keys():
            raise ValueError(f"data object has different keys than those specified in the spec: {spec}")
        for k in spec:
            if not matches_spec(o[k], spec[k], ignore_batch_dim=ignore_batch_dim):
                return False
            return True

    spec_shape = spec.shape[1:] if ignore_batch_dim else spec.shape
    o_shape = o.shape[1:] if ignore_batch_dim else o.shape
    return spec_shape == o_shape and spec.dtype == o.dtype
