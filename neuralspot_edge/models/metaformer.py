import keras
from pydantic import BaseModel, Field


class NameArgs(BaseModel):
    """Name and arguments"""

    name: str = Field(default="conv", description="Name")
    args: dict = Field(default_factory=dict, description="Arguments")


class MetaFormerBlockParams(BaseModel):
    """MetaFormer block parameters"""

    layers: int = Field(default=2, description="Number of layers")
    patch_embed: dict = Field(default_factory=dict, description="Patch embedding")
    token_mixer: NameArgs = Field(default_factory=dict, description="Token mixer")
    channel_mixer: NameArgs = Field(default_factory=dict, description="Channel mixer")


class MetaFormerParams(BaseModel):
    """MetaFormer parameters"""

    blocks: list[MetaFormerBlockParams] = Field(default_factory=list, description="MetaFormer blocks")
    output_filters: int = Field(default=0, description="Output filters")
    output_activation: str | None = Field(default=None, description="Output activation")
    include_top: bool = Field(default=True, description="Include top")
    dropout: float = Field(default=0.2, description="Dropout rate")
    drop_connect_rate: float = Field(default=0.2, description="Drop connect rate")
    name: str = Field(default="MetaFormer", description="Model name")


def patch_embedding(
    embed_dim: int,
    patch_shape: tuple[int, int],
    stride_shape: tuple[int, int] | None = None,
    padding: str = "same",
) -> keras.layers.Layer:
    """Patch embedding layer using 2D convolution"""
    # SHAPE (B, F, T, C) -> (B, F//S, T//S, E)
    # SHAPE (B, H, W, C) -> (B, H//S, W//S, E
    return keras.layers.Conv2D(
        filters=embed_dim,
        kernel_size=patch_shape,
        strides=stride_shape or patch_shape,
        padding=padding,
        use_bias=False,
    )


def pool_token_mixer(
    pool_size: tuple[int, int] = (2, 2),
) -> keras.layers.Layer:
    """Token mixer using average pooling

    Args:
        pool_size (tuple[int, int], optional): Pool size. Defaults to (2, 2).

    Returns:
        keras.layers.Layer: Token mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = (
            keras.layers.AveragePooling2D(
                pool_size=pool_size,
                strides=(1, 1),
                padding="same",
            )(y)
            - y
        )
        return y

    # END DEF
    return layer


def conv_token_mixer(
    embed_dim: int,
    kernel_size: tuple[int, int] = (3, 3),
    strides: tuple[int, int] = (1, 1),
):
    """Token mixer using separable convolution

    Args:
        embed_dim (int): Embedding dimension
        kernel_size (tuple[int, int], optional): Kernel size. Defaults to (3, 3).
        strides (tuple[int, int], optional): Strides. Defaults to (1, 1).

    Returns:
        keras.layers.Layer: Token mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = keras.layers.SeparableConv2D(
            filters=embed_dim,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            use_bias=False,
        )(y)
        return y

    # END DEF
    return layer


def attention_token_mixer(
    embed_dim: int,
    num_heads: int,
    dropout: float = 0.1,
) -> keras.layers.Layer:
    """Token mixer using multi-head attention

    Args:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of heads
        dropout (float, optional): Dropout rate. Defaults to 0.1.

    Returns:
        keras.layers.Layer: Token mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        input_shape = keras.ops.shape(x)
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        y = x  # Shape is B, H, W, C
        y = keras.layers.Reshape((height * width, channels))(y)
        y = keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout,
        )(y, y)
        # Reshape back to B, H, W, C
        y = keras.layers.Reshape((height, width, channels))(y)
        return y

    # END DEF
    return layer


def mlp_channel_mixer(
    embed_dim: int,
    ratio: int = 4,
    activation: str = "gelu",
    dropout: float = 0,
):
    """Channel mixer using MLP via 1x1 convolutions

    Args:
        embed_dim (int): Embedding dimension
        ratio (int, optional): Expansion ratio. Defaults to 4.
        activation (str, optional): Activation function. Defaults to "gelu".
        dropout (float, optional): Dropout rate. Defaults to 0.

    Returns:
        keras.layers.Layer: Channel mixer layer
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        y = keras.layers.Conv2D(
            filters=int(embed_dim * ratio),
            kernel_size=(1, 1),
            activation=activation,
        )(y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout)(y)

        y = keras.layers.Conv2D(
            filters=embed_dim,
            kernel_size=(1, 1),
        )(y)

        if 0 < dropout < 1:
            y = keras.layers.Dropout(dropout)(y)

        return y

    # END DEF
    return layer


def metaformer_block(
    token_mixer: keras.layers.Layer | None = None,
    channel_mixer: keras.layers.Layer | None = None,
    name: str = "mf_block",
) -> keras.layers.Layer:
    """Metaformer block

    Args:
        token_mixer (keras.layers.Layer, optional): Token mixer layer. Defaults to None.
        channel_mixer (keras.layers.Layer, optional): Channel mixer layer. Defaults to None.
        name (str, optional): Block name. Defaults to 'mf_block'.

    Returns:
        keras.layers.Layer: Metaformer block
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        # Shape is B, T, F, C
        y = x

        if token_mixer:
            # Apply layer normalization
            y1 = keras.layers.LayerNormalization(name=f"{name}.token_ln", axis=-1)(y)

            # Apply token mixer
            y1 = token_mixer(y1)

            # Add residual connection
            y = y + y1
        # END IF

        if channel_mixer:
            # Apply layer normalization
            y1 = keras.layers.LayerNormalization(name=f"{name}.ch_ln", axis=-1)(y)

            # Apply channel mixer
            y1 = channel_mixer(y1)

            # Add residual connection
            y = y + y1

        return y

    # END DEF

    return layer


def MetaFormer(
    x: keras.KerasTensor,
    params: MetaFormerParams,
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

    for b, block in enumerate(params.blocks):
        print("Adding block", b + 1)
        # Apply patch embedding
        y = patch_embedding(**block.patch_embed)(y)
        for lyr in range(block.layers):
            token_mixer = None
            if block.token_mixer.name == "conv":
                token_mixer = conv_token_mixer(**block.token_mixer.args)
            elif block.token_mixer.name == "pool":
                token_mixer = pool_token_mixer(**block.token_mixer.args)
            elif block.token_mixer.name == "attention":
                token_mixer = attention_token_mixer(**block.token_mixer.args)
            # END IF
            channel_mixer = None
            if block.channel_mixer.name == "mlp":
                channel_mixer = mlp_channel_mixer(**block.channel_mixer.args)
            # END IF
            y = metaformer_block(
                token_mixer=token_mixer,
                channel_mixer=channel_mixer,
                name=f"B{b+1}.L{lyr+1}",
            )(y)
        # END FOR
    # END FOR

    if params.include_top:
        name = "top"
        y = keras.layers.GlobalAveragePooling2D(name=f"{name}.gap")(y)
        if 0 < params.dropout < 1:
            y = keras.layers.Dropout(params.dropout, name=f"{name}.dropout")(y)
        if num_classes is not None:
            y = keras.layers.Dense(num_classes, name=name)(y)
        if params.output_activation:
            y = keras.layers.Activation(params.output_activation)(y)
    # END IF

    model = keras.Model(x, y, name=params.name)
    return model


def ccaa_metaformer(
    x: keras.KerasTensor,
    num_classes: int | None = None,
) -> keras.Model:
    """CCAA Metaformer model"""

    params = MetaFormerParams(
        blocks=[
            MetaFormerBlockParams(
                layers=2,
                patch_embed=dict(
                    embed_dim=32,
                    patch_shape=(7, 7),
                    stride_shape=(4, 4),
                ),
                token_mixer=dict(
                    name="conv",
                    args=dict(
                        embed_dim=32,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                    ),
                ),
                channel_mixer=dict(
                    name="mlp",
                    args=dict(
                        embed_dim=32,
                        ratio=4,
                        activation="gelu",
                        dropout=0.1,
                    ),
                ),
            ),
            MetaFormerBlockParams(
                layers=2,
                patch_embed=dict(
                    embed_dim=64,
                    patch_shape=(3, 3),
                    stride_shape=(2, 2),
                ),
                token_mixer=dict(
                    name="conv",
                    args=dict(
                        embed_dim=64,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                    ),
                ),
                channel_mixer=dict(
                    name="mlp",
                    args=dict(
                        embed_dim=64,
                        ratio=4,
                        activation="gelu",
                        dropout=0.1,
                    ),
                ),
            ),
            MetaFormerBlockParams(
                layers=2,
                patch_embed=dict(
                    embed_dim=128,
                    patch_shape=(3, 3),
                    stride_shape=(2, 2),
                ),
                token_mixer=dict(
                    name="conv",
                    args=dict(
                        embed_dim=128,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                    ),
                ),
                channel_mixer=dict(
                    name="mlp",
                    args=dict(
                        embed_dim=128,
                        ratio=4,
                        activation="gelu",
                        dropout=0.1,
                    ),
                ),
            ),
            MetaFormerBlockParams(
                layers=2,
                patch_embed=dict(
                    embed_dim=256,
                    patch_shape=(3, 2),
                    stride_shape=(3, 2),
                ),
                token_mixer=dict(
                    name="conv",
                    args=dict(
                        embed_dim=256,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                    ),
                ),
                channel_mixer=dict(
                    name="mlp",
                    args=dict(
                        embed_dim=256,
                        ratio=4,
                        activation="gelu",
                        dropout=0.1,
                    ),
                ),
            ),
        ],
        include_top=True,
        output_activation="softmax",
    )

    return MetaFormer(
        x=x,
        params=params,
        num_classes=num_classes,
    )


def metaformer_from_object(
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

    return MetaFormer(x=x, params=MetaFormerParams(**params), num_classes=num_classes)
