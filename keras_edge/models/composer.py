"""Composer network"""

import logging
import keras
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, se_block
from .activations import relu6
from .utils import load_model

logger = logging.getLogger(__name__)


class ComposerLayerParams(BaseModel):
    """Composer layer parameters"""

    name: str = Field(..., description="Layer name")
    params: dict = Field(default_factory=dict, description="Layer arguments")


class ComposerParams(BaseModel):
    """Composer Network parameters"""

    layers: list[ComposerLayerParams] = Field(default_factory=list, description="Network layers")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    name: str = Field(default="Composer", description="Model name")


def Composer(
    x: keras.KerasTensor,
    params: ComposerParams,
    num_classes: int | None = None,
) -> keras.Model:
    """Composes a sequential set of networks/layers.
    Useful for adding custom layers to a pre-trained model (e.g. foundation).
    """
    y = x
    for layer in params.layers:
        match layer.name:
            case "conv2d":
                y = conv2d(y, **layer.params)
            case "dense":
                y = keras.layers.Dense(**layer.params)(y)
            case "relu6":
                y = relu6(y)
            case "batch_norm":
                y = batch_norm(y)
            case "se_block":
                y = se_block(y, **layer.params)
            case "load_model":
                prev_model = load_model(layer.params["model_file"])
                trainable = layer.params.get("trainable", True)
                if not trainable:
                    logger.info(f"Freezing model {prev_model.name}")
                prev_model.trainable = trainable
                y = prev_model(y, training=trainable)
            case _:
                raise ValueError(f"Unknown layer {layer.name}")
        # END MATCH
    # END FOR

    if params.include_top:
        if num_classes is not None:
            y = keras.layers.Dense(num_classes)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    model = keras.Model(x, y, name=params.name)
    return model


def composer_from_object(x: keras.KerasTensor, params: dict, num_classes: int | None = None) -> keras.Model:
    """Create model from object

    Args:
        x (keras.KerasTensor): Input tensor
        params (dict): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    return Composer(x=x, params=ComposerParams(**params), num_classes=num_classes)
