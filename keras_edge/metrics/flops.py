import os

import keras
import tensorflow as tf

# pylint: disable=no-name-in-module
from tensorflow.python.profiler.model_analyzer import profile

# pylint: disable=no-name-in-module
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder


def get_flops(model: keras.Model, batch_size: int | None = None, fpath: os.PathLike | None = None) -> float:
    """Calculate FLOPS for keras.Model or keras.Sequential.
    Ignore operations used in only training mode such as Initialization.
    Use tf.profiler of tensorflow v2 api.

    Known Limitations: Does not support LSTM and GRU.

    Args:
        model (keras.Model|keras.Sequential): Model
        batch_size (int, optional): Batch size. Defaults to None.
        fpath (os.PathLike, optional): Output file path. Defaults to None.

    Returns:
        float: FLOPS
    """
    input_signature = [tf.TensorSpec(shape=(batch_size,) + model.input_shape[1:])]
    forward_pass = tf.function(model.call, input_signature=input_signature)
    graph = forward_pass.get_concrete_function().graph
    options = ProfileOptionBuilder.float_operation()
    if fpath:
        options["output"] = f"file:outfile={fpath}"
    graph_info = profile(graph, options=options)
    return float(graph_info.total_float_ops)
