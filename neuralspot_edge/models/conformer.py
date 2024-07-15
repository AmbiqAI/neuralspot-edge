"""Fast Conformer"""

import keras
from pydantic import BaseModel, Field

from .blocks import layer_norm, batch_norm
from .activations import swish, glu, relu


class SubsampleBlockParams(BaseModel):
    depth: int = 256
    kernel_size: int = 3
    strides: int = 2


class ConformerBlockParams(BaseModel):
    depth: int = 256
    fc_ex_factor: float = 4
    fc_res_factor: float = 0.5
    embedding: str = "relative"
    num_heads: int = 4
    kernel_size: int = 9
    dropout: float = 0.1
    use_bias: bool = True


class ConformerParams(BaseModel):
    """Conformer parameters"""

    subsamples: list[SubsampleBlockParams] = Field(default_factory=list, description="Subsample blocks")
    blocks: list[ConformerBlockParams] = Field(default_factory=list, description="Conformer blocks")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    name: str = Field(default="Fast Conformer", description="Model name")


def subsampler(
    blocks: SubsampleBlockParams,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    name: str | None = None,
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        # Apply subsampling blocks
        for i, block in enumerate(blocks):
            y = keras.layers.SeparableConv2D(
                filters=block.depth,
                kernel_size=block.kernel_size,
                strides=block.strides,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                depthwise_regularizer=kernel_regularizer,
                pointwise_regularizer=kernel_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                padding="same",
                name=f"{name}_conv{i+1}",
            )(y)
            y = relu(name=f"{name}_relu{i+1}")(y)
        # END FOR

        # Swap from (b,f,t,c) to (b,t,c,f) and merge (c,f) to get (b,t,c*f)
        y = keras.ops.transpose(y, axes=[0, 2, 3, 1])
        y = keras.layers.Reshape((y.shape[1], -1))(y)

        # Can serve as learnable embedding
        y = keras.layers.Dense(
            units=blocks[-1].depth,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f"{name}_dense",
        )(y)  # (b,t,c*f) -> (b,t,d)
        return y

    # END DEF
    return layer


def fc_block(
    depth: int,
    ex_factor: int = 4,
    residual_factor: float = 0.5,
    dropout: float = 0,
    use_bias: bool = True,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    name: str = "fc_block",
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x

        y = layer_norm(name=f"{name}.ln")(y)

        y = keras.layers.Dense(
            int(ex_factor * depth),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_bias=use_bias,
            name=f"{name}.fc1",
        )(y)
        y = swish(name=f"{name}.swish")(y)
        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout, name=f"{name}.dropout1")(y)
        # END IF
        y = keras.layers.Dense(
            depth,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            use_bias=use_bias,
            name=f"{name}.fc2",
        )(y)
        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout, name=f"{name}.dropout2")(y)
        # END IF
        return x + residual_factor * y

    # END DEF
    return layer


def conv_block(
    depth: int,
    kernel_size: int = 9,
    dropout: float = 0.0,
    padding: str = "same",
    scale_factor: int = 2,
    kernel_initializer: str = "glorot_uniform",
    bias_initializer: str = "zeros",
    kernel_regularizer=None,
    bias_regularizer=None,
    name: str = "conv_module",
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x

        y = layer_norm(name=f"{name}.ln")(y)

        y = keras.layers.Conv1D(
            filters=scale_factor * depth,
            kernel_size=1,
            strides=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}.pw_conv1",
        )(y)

        y = glu(name=f"{name}.glu")(y)

        y = keras.layers.DepthwiseConv1D(
            kernel_size=kernel_size,
            strides=1,
            padding=padding,
            depthwise_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}.dw_conv",
        )(y)
        y = batch_norm(name=f"{name}.bn")(y)

        y = swish(name=f"{name}.swish")(y)

        y = keras.layers.Conv1D(
            filters=depth,
            kernel_size=1,
            strides=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}.pw_conv2",
        )(y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(rate=dropout, name=f"{name}.dropout")(y)
        # END IF

        # Residual connection
        return x + y

    # END DEF
    return layer


def att_block(
    depth: int,
    embedding: str = "rel",
    num_heads: int = 4,
    dropout: float = 0.1,
    name: str = "att_block",
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = layer_norm(name=f"{name}.ln")(y)

        # Att type: rel, abs, none
        y = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=depth // num_heads,
            value_dim=depth // num_heads,
            dropout=dropout,
            name=f"{name}.mha",
        )(y, y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout, name=f"{name}.dropout")(y)

        return y

    # END DEF
    return layer


def conformer_block(
    depth: int,
    fc_ex_factor: int = 4,
    fc_res_factor: int = 0.5,
    embedding: str = "relative",
    num_heads: int = 4,
    kernel_size: int = 9,
    dropout: float = 0.1,
    use_bias: bool = True,
    name: str = "cf_block",
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x

        # 1st FC Block
        y = fc_block(
            depth=depth,
            ex_factor=fc_ex_factor,
            residual_factor=fc_res_factor,
            dropout=dropout,
            use_bias=use_bias,
            name=f"{name}.fcb1",
        )(y)

        # ATT Block w/ residual
        y = att_block(
            depth=depth,
            embedding=embedding,
            num_heads=num_heads,
            dropout=dropout,
            name=f"{name}.att",
        )(y)

        # Conv Block w/ residual
        y = conv_block(depth=depth, kernel_size=kernel_size, dropout=dropout, name=f"{name}.cvb")(y)

        # 2nd FC Block w/ residual
        y = fc_block(
            depth=depth,
            ex_factor=fc_ex_factor,
            residual_factor=fc_res_factor,
            dropout=dropout,
            use_bias=use_bias,
            name=f"{name}.fcb2",
        )(y)

        # Output Layer Norm
        y = layer_norm(name=f"{name}.out.ln")(y)

        return y

    # END DEF
    return layer


def Conformer(
    x: keras.KerasTensor,
    params: ConformerParams,
    num_classes: int | None = None,
) -> keras.Model:
    """MetaFormer model

    Args:
        x (keras.KerasTensor): Input tensor
        params (MetaFormerParams): Model parameters.
        num_classes (int, optional): # classes.

    Returns:
        keras.Model: Model
    """
    y = x

    y = subsampler(
        blocks=params.subsamples,
        kernel_size=params.subsample.kernel_size,
        strides=params.subsample.strides,
        num_downsamples=params.subsample.downsamples,
        name="subsample",
    )(y)

    for i, block in enumerate(params.blocks):
        y = conformer_block(
            depth=block.depth,
            fc_ex_factor=block.fc_ex_factor,
            fc_res_factor=block.fc_res_factor,
            embedding=block.embedding,
            num_heads=block.num_heads,
            kernel_size=block.kernel_size,
            dropout=block.dropout,
            use_bias=block.use_bias,
            name=f"CB{i+1}",
        )(y)

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling1D(name=f"{name}.gap")(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes, name=name)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)
    # END IF

    model = keras.Model(x, y, name=params.name)
    return model


def conformer_from_object(
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
    return Conformer(x=x, params=ConformerParams(**params), num_classes=num_classes)
