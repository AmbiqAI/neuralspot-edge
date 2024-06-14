""" Fast Conformer"""
import keras
from pydantic import BaseModel, Field

from .blocks import swish, glu, layer_norm, batch_norm, relu
from .defines import MBConvParams
from .utils import make_divisible

class ConformerParams(BaseModel):
    """Conformer parameters"""
    pass

def subsample_block(
    filters: int,
    depth: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 2,
    num_downsamples: int = 3,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    name: str | None = None,
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_initializer=bias_initializer,
            bias_regularizer=bias_regularizer,
            padding='same',
            name=f'{name}_conv1'
        )
        y = relu()(y)

        # Apply convolutional downsampling
        for i in range(1, num_downsamples):
            y = keras.layers.SeparableConv2D(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                depthwise_initializer=kernel_initializer,
                pointwise_initializer=kernel_initializer,
                depthwise_regularizer=kernel_regularizer,
                pointwise_regularizer=kernel_regularizer,
                bias_initializer=bias_initializer,
                bias_regularizer=bias_regularizer,
                padding='same',
                name=f'{name}_conv{i}'
            )
            y = relu()(y)
        # END FOR

        # Swap to (b,t,c,f) and merge (c,f)
        keras.ops.transpose(y, perm=[0, 1, 3, 2])
        y = keras.layers.Reshape((y.shape[0], y.shape[1], -1))(y)

        y = keras.layers.Dense(
            units=depth,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=f'{name}_dense'
        )(y)
        return y
    # END DEF
    return layer

def fc_block(
    depth: int,
    fc_depth: int = 4,
    dropout: float = 0,
    use_bias: bool = True,
    name: str = 'fc_block',
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = keras.layers.Dense(
            fc_depth,
            use_bias=use_bias,
            name=f"{name}.fc1"
            )(y)
        y = swish(
            name=f"{name}.swish"
        )(y)
        if dropout > 0:
            y = keras.layers.Dropout(
                dropout,
                name=f"{name}.dropout"
            )(y)
        y = keras.layers.Dense(
            depth,
            use_bias=use_bias,
            name=f"{name}.fc2"
            )(y)
        )(y)
        return y
    # END DEF
    return layer

def conv_block(
    depth: int,
    kernel_size: int = 9,
    dropout: float = 0.0,
    padding: str = "causal",
    scale_factor: int = 2,
    kernel_initializer=None,
    bias_initializer=None,
    kernel_regularizer=None,
    bias_regularizer=None,
    norm_type='batch_norm',
    name: str = "conv_module",
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x

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
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            name=f"{name}.dw_conv"
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
            name=f"{name}.pw_conv2"
        )(y)

        y = keras.layers.Dropout(
            rate=dropout,
            name=f"{name}.dropout"
        )(y)

        # Residual connection
        y = keras.layers.Add()([x, y])

        return y
    # END DEF
    return layer

def att_block(
    att_type: str = "relative",
    num_heads: int = 4,
    dropout: float = 0.1,
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        return y
    # END DEF
    return layer

def conformer_block(
    depth: int,
    fc_depth: int,
    fc_factor: int = 0.5,
    att_dropout=0.1,
    att_type: str = "relative",
    att_num_heads: int = 4,
    conv_kernel_size: int = 9,
    conv_norm_type: str = "batch",
    dropout: float = 0.1,
    use_bias: bool = True,
    name: str = 'cf_block',
):
    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        residual = y

        y = layer_norm(
            name=f"{name}.fcb1_pre_ln"
        )(y)
        y = fc_block(
            depth=depth,
            fc_depth=fc_depth,
            dropout=dropout,
            use_bias=use_bias,
            name=f"{name}.fcb1"
        )(y)
        y = keras.layers.Dropout(
            dropout,
            name=f"{name}.fcb1_post_dropout"
        )(y)
        residual = residual + y * fc_factor

        y = layer_norm(
            name=f"{name}.att_pre_ln"
        )(residual)
        y = att_block()(y)
        y = keras.layers.Dropout(
            dropout,
            name=f"{name}.att_post_dropout"
        )(y)
        residual = residual + y

        y = layer_norm(
            name=f"{name}.cvb_pre_ln"
        )(residual)
        y = conv_block(
            depth=depth,
            kernel_size=conv_kernel_size,
            dropout=dropout,
            norm_type=conv_norm_type,
            name=f"{name}.cvb"
        )(y)
        y = keras.layers.Dropout(
            dropout,
            name=f"{name}.cvb_post_dropout"
        )(y)

        residual = residual + y

        y = layer_norm(
            name=f"{name}.fcb2_pre_ln"
        )(residual)
        y = fc_block(
            depth=depth,
            fc_depth=fc_depth,
            dropout=dropout,
            use_bias=use_bias,
            name=f"{name}.fcb2"
        )(y)
        y = keras.layers.Dropout(
            dropout,
            name=f"{name}.fcb2_post_dropout"
        )(y)

        residual = residual + y * fc_factor
        y = layer_norm(
            name=f"{name}.out_ln"
        )(residual)

        return y
