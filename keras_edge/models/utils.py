
def make_divisible(v: int, divisor: int = 4, min_value: int | None = None) -> int:
    """Ensure layer has # channels divisble by divisor
       https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py

    Args:
        v (int): # channels
        divisor (int, optional): Divisor. Defaults to 4.
        min_value (int | None, optional): Min # channels. Defaults to None.

    Returns:
        int: # channels
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
