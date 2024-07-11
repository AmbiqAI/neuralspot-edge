from collections.abc import Iterable
import keras
from .activations import relu6, sigmoid


def layer_norm(name: str | None = None, axis=-1, scale: bool = True) -> keras.Layer:
    """Layer normalization layer"""
    name = name + ".ln" if name else None
    return keras.layers.LayerNormalization(axis=axis, name=name, scale=scale)


def batch_norm(name: str | None = None, momentum=0.9, epsilon=1e-3, axis: int = -1) -> keras.Layer:
    """Batch normalization layer"""
    name = name + ".bn" if name else None
    return keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, axis=axis, name=name)


def norm_layer(norm: str, name: str) -> keras.Layer:
    """Normalization layer

    Args:
        norm (str): Normalization type
        name (str): Name

    Returns:
        KerasLayer: Layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        """Functional normalization layer

        Args:
            x (tf.Tensor): Input tensor

        Returns:
            tf.Tensor: Output tensor
        """
        if norm == "batch":
            return batch_norm(name=name)(x)
        if norm == "layer":
            return layer_norm(name=name)(x)
        return x

    return layer


def conv2d(
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    padding: str = "same",
    use_bias: bool = False,
    groups: int = 1,
    dilation: int = 1,
    name: str | None = None,
) -> keras.Layer:
    """2D convolutional layer

    Args:
        filters (int): # filters
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        use_bias (bool, optional): Add bias. Defaults to False.
        groups (int, optional): # groups. Defaults to 1.
        dilation (int, optional): Dilation rate. Defaults to 1.
        name (str | None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional 2D conv layer
    """
    name = name + ".conv" if name else None
    return keras.layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups,
        dilation_rate=dilation,
        kernel_initializer="he_normal",
        name=name,
    )


def conv1d(
    filters: int,
    kernel_size: int = 3,
    strides: int = 1,
    padding: str = "same",
    use_bias: bool = False,
    name: str | None = None,
) -> keras.Layer:
    """1D convolutional layer using 2D convolutional layer

    Args:
        filters (int): # filters
        kernel_size (int, optional): Kernel size. Defaults to 3.
        strides (int, optional): Stride length. Defaults to 1.
        padding (str, optional): Padding. Defaults to "same".
        use_bias (bool, optional): Add bias. Defaults to False.
        name (str | None, optional): Layer name. Defaults to None.

    Returns:
        keras.Layer: Functional 1D conv layer
    """
    name = name + ".conv" if name else None
    return keras.layers.Conv2D(
        filters,
        kernel_size=(1, kernel_size),
        strides=(1, strides),
        padding=padding,
        use_bias=use_bias,
        kernel_initializer="he_normal",
        name=name,
    )


def se_block(
    ratio: int = 8,
    name: str | None = None,
    activation: str = "relu6",
) -> keras.Layer:
    """Squeeze & excite block

    Args:
        ratio (Expansion ratio, optional): Expansion ratio. Defaults to 8.
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        keras.Layer: Functional SE layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        num_chan = x.shape[-1]
        # Squeeze
        name_pool = f"{name}.pool" if name else None
        name_sq = f"{name}.sq" if name else None
        name_act = f"{name}.sq.act" if name else None
        y = keras.layers.GlobalAveragePooling2D(name=name_pool, keepdims=True)(x)
        y = conv2d(int(num_chan // ratio), kernel_size=(1, 1), use_bias=True, name=name_sq)(y)
        y = keras.layers.Activation(activation, name=name_act)(y)
        # Excite
        name_ex = f"{name}.ex" if name else None
        y = conv2d(num_chan, kernel_size=(1, 1), use_bias=True, name=name_ex)(y)
        y = sigmoid(name=name_ex, hard=True)(y)
        y = keras.layers.Multiply()([x, y])
        return y

    # END DEF

    return layer


def mbconv_block(
    output_filters: int,
    expand_ratio: float = 1,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    se_ratio: float = 8,
    droprate: float = 0,
    norm: str = "batch",
    activation: str = "relu6",
    name: str | None = None,
) -> keras.Layer:
    """MBConv block w/ expansion and SE

    Args:
        output_filters (int): # output filter channels
        expand_ratio (float, optional): Expansion ratio. Defaults to 1.
        kernel_size (int | tuple[int, int], optional): Kernel size. Defaults to 3.
        strides (int | tuple[int, int], optional): Stride length. Defaults to 1.
        se_ratio (float, optional): SE ratio. Defaults to 8.
        droprate (float, optional): Drop rate. Defaults to 0.
        norm (str, optional): Normalization type. Defaults to "batch".
        activation (str, optional): Activation function. Defaults to "relu6".
        name (str|None, optional): Block name. Defaults to None.

    Returns:
        keras.Layer: Functional layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_filters = x.shape[-1]
        stride_len = strides if isinstance(strides, int) else sum(strides) / len(strides)
        is_symmetric = isinstance(kernel_size, Iterable) and kernel_size[0] == kernel_size[1]
        is_downsample = not is_symmetric and stride_len > 1

        add_residual = input_filters == output_filters and stride_len == 1
        # Expand: narrow -> wide
        if expand_ratio != 1:
            name_ex = f"{name}.exp" if name else None
            filters = int(input_filters * expand_ratio)
            y = conv2d(filters, kernel_size=(1, 1), strides=(1, 1), name=name_ex)(x)
            y = batch_norm(name=name_ex)(y)
            y = relu6(name=name_ex)(y)
        else:
            y = x

        # Apply: wide -> wide
        name_dp = f"{name}.dp" if name else None
        y = keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides if is_symmetric else (1, 1),
            padding="same",
            use_bias=False,
            depthwise_initializer="he_normal",
            name=name_dp,
        )(y)
        y = batch_norm(name=name_dp)(y)
        y = relu6(name=name_dp)(y)
        # NOTE: DepthwiseConv2D only supports equal size stride -> use maxpooling as needed
        if is_downsample:
            y = keras.layers.MaxPool2D(pool_size=strides, padding="same")(y)
        # END IF

        # SE: wide -> wide
        if se_ratio:
            name_se = f"{name}.se" if name else None
            y = se_block(ratio=se_ratio * expand_ratio, name=name_se)(y)

        # Reduce: wide -> narrow
        name_red = f"{name}.red" if name else None
        y = conv2d(
            output_filters,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            name=name_red,
        )(y)
        y = batch_norm(name=name_red)(y)
        # No activation

        if add_residual:
            name_res = f"{name}.res" if name else None
            if droprate > 0:
                y = keras.layers.Dropout(droprate, noise_shape=(None, 1, 1, 1))(y)
            y = keras.layers.add([x, y], name=name_res)
        return y

    # END DEF

    return layer
