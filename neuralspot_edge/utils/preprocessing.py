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

def convert_inputs_to_tf_dataset(x=None, y=None, sample_weight=None, batch_size=None):
    import tensorflow as tf

    if sample_weight is not None:
        raise ValueError(
            "Contrastive trainers do not yet support `sample_weight`."
        )

    if isinstance(x, tf.data.Dataset):
        if y is not None or batch_size is not None:
            raise ValueError(
                "When `x` is a `tf.data.Dataset`, please do not "
                "provide a value for `y` or `batch_size`. "
                "Got `y={y}`, `batch_size={batch_size}`."
            )
        return x

    # batch_size defaults to 32, as it does in fit().
    batch_size = batch_size or 32
    # Parse inputs
    inputs = x
    if y is not None:
        inputs = (x, y)

    # Construct tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    return dataset
