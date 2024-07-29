from typing import TypeVar

T = TypeVar("T")


def parse_factor(
    param: T | tuple[T, T], min_value: float = 0.0, max_value: float = 1.0, param_name: str = "factor"
) -> tuple[T, T]:
    if isinstance(param, (float, int)):
        param = (min_value, param)
    # END IF

    if param[0] is None:
        param[0] = param[1]
    # END IF

    if param[0] > param[1]:
        raise ValueError(
            f"`{param_name}[0] > {param_name}[1]`, `{param_name}[0]` must be "
            f"<= `{param_name}[1]`. Got `{param_name}={param}`"
        )
    if (min_value is not None and param[0] < min_value) or (max_value is not None and param[1] > max_value):
        raise ValueError(
            f"`{param_name}` should be inside of range " f"[{min_value}, {max_value}]. Got {param_name}={param}"
        )
    # END IF

    return param[0], param[1]
