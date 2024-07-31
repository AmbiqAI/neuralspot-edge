import random
import numpy as np
import keras

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
    keras.utils.set_random_seed(seed)

    return seed
