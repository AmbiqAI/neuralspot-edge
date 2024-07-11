"""RegNet https://arxiv.org/abs/2101.00590"""

from typing import Callable, Literal, cast

import keras
from pydantic import BaseModel, Field

from .blocks import batch_norm, conv2d, se_block
from .activations import relu6
from .utils import make_divisible


class RegNetBlockParam(BaseModel):
    """RegNet block parameters"""

    filters: int = Field(..., description="# filters")
    depth: int = Field(default=1, description="Layer depth")
    group_width: int = Field(default=1, description="Group width. Must be divisible by in/out filters")
    kernel_size: int | tuple[int, int] = Field(default=3, description="Kernel size")
    strides: int | tuple[int, int] = Field(default=1, description="Stride size")
    se_ratio: float = Field(default=8, description="Squeeze Excite ratio")
    droprate: float = Field(default=0, description="Drop rate")


class RegNetParams(BaseModel):
    """RegNet parameters"""

    blocks: list[RegNetBlockParam] = Field(default_factory=list, description="RegNet blocks")
    input_filters: int = Field(default=0, description="Input filters")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    output_filters: int = Field(default=0, description="Output filters")
    block_style: Literal["y", "z"] = Field(default="y", description="Block style")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    dropout: float = Field(default=0.2, description="Dropout rate")
    name: str = Field(default="RegNet", description="Model name")


def yblock(
    output_filters: int = 0,
    group_width: int = 0,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 4,
    name: str | None = None,
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """RegNet Y-Block

    Args:
        output_filters (int, optional): # output filters. Defaults to 0.
        group_width (int, optional): Group width. Defaults to 0.
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 4.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        Callable[[keras.KerasTensor], keras.KerasTensor]: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters = x.shape[-1]
        groups = output_filters // group_width
        use_skip = (input_filters != output_filters) or (strides != 1 if isinstance(strides, int) else strides[0] != 1)

        name_ex = f"{name}.exp" if name else None
        y = conv2d(output_filters, kernel_size=(1, 1), name=name_ex)(x)
        y = batch_norm(name=name_ex)(y)
        y = relu6(name=name_ex)(y)

        y = cast(
            keras.KerasTensor,
            keras.layers.Conv2D(
                output_filters,
                kernel_size=kernel_size,
                strides=strides,
                groups=groups,
                padding="same",
                use_bias=False,
                kernel_initializer=keras.initializers.VarianceScaling(),  # type: ignore
            )(y),
        )
        y = batch_norm()(y)
        y = relu6()(y)

        if se_ratio:
            y = se_block(ratio=se_ratio)(y)

        y = conv2d(output_filters, kernel_size=(1, 1), strides=(1, 1))(y)
        y = batch_norm()(y)
        y = relu6()(y)

        if use_skip:
            x = conv2d(output_filters, kernel_size=(1, 1), strides=strides)(x)
            x = batch_norm()(x)

        y = cast(keras.KerasTensor, keras.layers.add([y, x]))
        y = relu6()(y)
        return y

    return layer


def zblock(
    output_filters: int = 0,
    group_width: int = 0,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 4,
    name: str | None = None,
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """RegNet X-Block

    Args:
        output_filters (int, optional): # output filters. Defaults to 0.
        group_width (int, optional): Group width. Defaults to 0.
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 4.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        Callable[[keras.KerasTensor], keras.KerasTensor]: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters = x.shape[-1]
        groups = input_filters // group_width
        use_add = input_filters == output_filters and (strides == 1 if isinstance(strides, int) else strides[0] == 1)
        expand_ratio = 2

        name_ex = f"{name}.exp" if name else None
        filters = input_filters * expand_ratio
        y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), name=name_ex)(x)
        y = batch_norm(name=name_ex)(y)
        y = relu6(name=name_ex)(y)

        y = cast(
            keras.KerasTensor,
            keras.layers.Conv2D(
                filters,
                kernel_size=kernel_size,
                strides=strides,
                groups=groups,
                padding="same",
                use_bias=False,
                kernel_initializer=keras.initializers.VarianceScaling(),  # type: ignore
            )(y),
        )
        y = batch_norm()(y)
        y = relu6()(y)

        if se_ratio:
            y = se_block(ratio=se_ratio)(y)

        y = conv2d(output_filters, kernel_size=(1, 1), strides=(1, 1))(y)
        y = batch_norm()(y)

        if use_add:
            y = cast(keras.KerasTensor, keras.layers.add([y, x]))

        return y

    return layer


def regnet_core(
    blocks: list[RegNetBlockParam],
    block_style: Literal["y", "z"] = "y",
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """RegNet core

    Args:
        blocks (list[RegNetBlockParam]): Block params
        block_style (float, optional): Block style. Defaults to 'y'.

    Returns:
        Callable[[keras.KerasTensor], keras.KerasTensor]: Core
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        block_fn = yblock if block_style == "y" else zblock
        for i, block in enumerate(blocks):
            filters = make_divisible(block.filters, 8)
            for d in range(block.depth):
                name = f"stage{i}.block{d+1}"
                x = block_fn(
                    output_filters=filters,
                    group_width=block.group_width,
                    kernel_size=block.kernel_size,
                    strides=block.strides,
                    se_ratio=block.se_ratio,
                    name=name,
                )(x)
            # END FOR
        # END FOR
        return x

    return layer


def RegNet(
    x: keras.KerasTensor,
    params: RegNetParams,
    num_classes: int | None = None,
):
    """Create RegNet TF functional model

    Args:
        x (keras.KerasTensor): Input tensor
        params (RegNetParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    # Stem
    if params.input_filters > 0:
        name = "stem"
        filters = make_divisible(params.input_filters, 8)
        y = conv2d(filters, kernel_size=(3, 3), strides=params.input_strides, name=name)(x)
        y = batch_norm(name=name)(y)
        y = cast(keras.KerasTensor, relu6(name=name)(y))
    else:
        y = x

    y = regnet_core(blocks=params.blocks, block_style=params.block_style)(y)

    if params.output_filters:
        name = "neck"
        filters = make_divisible(params.output_filters, 8)
        y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), padding="valid", name=name)(y)
        y = batch_norm(name=name)(y)
        y = relu6(name=name)(y)

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}.pool")(y)

        if params.dropout > 0 and params.dropout < 1:
            y = keras.layers.Dropout(params.dropout)(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes, name=name)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)

    model = keras.Model(x, y, name=params.name)
    return model


def regnet_from_object(
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
    return RegNet(x=x, params=RegNetParams(**params), num_classes=num_classes)
