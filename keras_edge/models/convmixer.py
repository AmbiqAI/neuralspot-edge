"""ConvMixer https://arxiv.org/abs/2201.09792"""

from typing import Callable

import keras
from pydantic import BaseModel, Field

from .blocks import batch_norm
from .activations import gelu


class ConvMixerParams(BaseModel):
    """ConvMixer parameters"""

    filters: int = Field(default=256, description="# filters per layer")
    depth: int = Field(default=8, description="Network depth")
    kernel_size: int = Field(default=5, description="Filter size")
    patch_size: int = Field(default=2, description="Patch size")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    name: str = Field(default="RegNet", description="Model name")


def conv_mixer_block(filters: int, kernel_size: int) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """ConvMixer block

    Args:
        filters (int): # filters
        kernel_size (int): Kernel size
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        # Depthwise convolution.
        x0 = x
        x = keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
        x = gelu()(x)
        x = batch_norm()(x)
        # Residual
        x = keras.layers.Add()([x, x0])

        # Pointwise convolution.
        x = keras.layers.Conv2D(filters, kernel_size=1)(x)
        x = gelu()(x)
        x = batch_norm()(x)
        return x

    return layer


def ConvMixer(
    x: keras.KerasTensor,
    params: ConvMixerParams,
    num_classes: int | None = None,
):
    """ConvMixer: https://openreview.net/pdf?id=TVHS5Y4dNvM.

    Args:
        x (keras.KerasTensor): Input tensor
        params (ConvMixerParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    # Extract patch embeddings
    y = keras.layers.Conv2D(
        filters=params.filters,
        kernel_size=params.patch_size,
        strides=params.patch_size,
        padding="same",
        use_bias=True,
    )(x)
    y = gelu()(y)
    y = batch_norm()(y)

    # ConvMixer blocks
    for _ in range(params.depth):
        y = conv_mixer_block(filters=params.filters, kernel_size=params.kernel_size)(y)

    # Classification block
    if params.include_top:
        y = keras.layers.GlobalAvgPool2D(keepdims=False)(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    model = keras.Model(x, y)
    return model


def convmixer_from_object(
    x: keras.KerasTensor,
    params: dict,
    num_classes: int | None = None,
) -> keras.Model:
    """Create model from object

    Args:
        x (keras.KerasTensor): Input tensor
        params (dict): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    return ConvMixer(x=x, params=ConvMixerParams(**params), num_classes=num_classes)
