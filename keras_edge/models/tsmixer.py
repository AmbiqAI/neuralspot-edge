"""Implementation of TSMixer."""

from typing import Literal

import keras
from pydantic import BaseModel, Field


class TsBlockParams(BaseModel):
    """TsMixer block parameters"""

    norm: Literal["batch", "layer"] | None = Field(default="layer", description="Normalization type")
    activation: Literal["relu", "gelu"] | None = Field(default="relu", description="Activation type")
    dropout: float | None = Field(default=None, description="Dropout rate")
    ff_dim: int | None = Field(default=None, description="Feed forward dimension")


class TsMixerParams(BaseModel):
    """TsMixer parameters"""

    blocks: list[TsBlockParams] = Field(default_factory=list, description="UNext blocks")
    name: str = Field(default="TsMixer", description="Model name")


def norm_layer(norm: str, name: str) -> keras.Layer:
    """Normalization layer

    Args:
        norm (str): Normalization type
        name (str): Name

    Returns:
        keras.Layer: Layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """Functional normalization layer

        Args:
            x (keras.KerasTensor): Input tensor

        Returns:
            keras.KerasTensor: Output tensor
        """
        if norm == "batch":
            return keras.layers.BatchNormalization(axis=[-2, -1], name=f"{name}.BN")(x)
        if norm == "layer":
            return keras.layers.LayerNormalization(axis=[-2, -1], name=f"{name}.LN")(x)
        return x

    return layer


def ts_block(params: TsBlockParams, name: str) -> keras.Layer:
    """Residual block of TSMixer."""

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        # Temporal Linear
        y = norm_layer(params.norm, name=f"{name}.TL")(x)

        y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Channel, Input Length]
        y = keras.layers.Dense(y.shape[-1], activation=params.activation, name=f"{name}.TL.DENSE")(y)
        y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Input Length, Channel]
        y = keras.layers.Dropout(params.dropout, name=f"{name}.TL.DROP")(y)
        res = y + x

        # Feature Linear
        y = norm_layer(params.norm, name=f"{name}.FL")(res)
        y = keras.layers.Dense(params.ff_dim, activation=params.activation, name=f"{name}.FL.DENSE")(
            y
        )  # [Batch, Input Length, FF_Dim]
        y = keras.layers.Dropout(params.dropout, name=f"{name}.FL.DROP")(y)

        y = keras.layers.Dense(x.shape[-1], name=f"{name}.RL.DENSE")(y)  # [Batch, Input Length, Channel]
        y = keras.layers.Dropout(params.dropout, name=f"{name}.RL.DROP")(y)
        return y + res

    return layer


def TsMixer(x: keras.KerasTensor, params: any, num_classes: int):
    y = x
    for block in range(params.blocks):
        y = ts_block(block)(y)

    # if target_slice:
    #     y = y[:, :, target_slice]

    y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Channel, Input Length]
    y = keras.layers.Dense(num_classes)(y)  # [Batch, Channel, Output Length]
    y = keras.ops.transpose(y, axes=[0, 2, 1])  # [Batch, Output Length, Channel])

    # Define the model
    model = keras.Model(x, y, name=params.name)
    return model


def tsmixer_from_object(
    x: keras.KerasTensor,
    params: dict,
    num_classes: int,
) -> keras.Model:
    """Create model from object

    Args:
        x (keras.KerasTensor): Input tensor
        params (dict): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    return TsMixer(x=x, params=TsMixerParams(**params), num_classes=num_classes)
