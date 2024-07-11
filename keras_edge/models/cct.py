from typing import Callable, cast

import numpy as np
import keras


def cct_tokenizer_block(
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    pooling_kernel_size: int = 3,
    pooling_stride: int = 2,
    num_conv_layers: int = 2,
    filter_sizes: tuple[int, int] = (64, 128),
) -> Callable[[keras.KerasTensor], keras.KerasTensor]:
    """CCT tokenizer block

    Args:
        kernel_size (int, optional): Kernel size. Defaults to 3.
        stride (int, optional): Stride length. Defaults to 1.
        padding (int, optional): Padding. Defaults to 1.
        pooling_kernel_size (int, optional): Pooling kernel size. Defaults to 3.
        pooling_stride (int, optional): Pooling stride. Defaults to 2.
        num_conv_layers (int, optional): # conv layers. Defaults to 2.
        filter_sizes (tuple[int], optional): # filters per layer. Defaults to (64, 128).
    """

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        y = x
        for filter_size in filter_sizes:
            y = keras.layers.Conv2D(
                filter_size,
                kernel_size,
                stride,
                padding="valid",
                use_bias=False,
                activation="relu",
                kernel_initializer="he_normal",
            )(y)
            y = keras.layers.ZeroPadding2D(padding)(y)
            y = keras.layers.MaxPool2D(pooling_kernel_size, pooling_stride, "same")(y)
        # END FOR

        y = cast(
            keras.KerasTensor,
            keras.layers.Reshape((-1, y.shape[1] * y.shape[2], y.shape[-1]))(y),
        )
        return y

    return layer


class StochasticDepth(keras.layers.Layer):
    """StochasticDepth"""

    def __init__(self, drop_prop: float, **kwargs):
        """Stochastic Depth
        Args:
            drop_prop (float): Drop probability
        """
        super().__init__(**kwargs)
        self.drop_prob = drop_prop

    def call(self, inputs, *args, **kwargs):
        """Forward pass

        Args:
            x (keras.KerasTensor): Input tensor
            training (bool, optional): If training mode. Defaults to None.

        Returns:
            keras.KerasTensor: Output tensor
        """
        if kwargs.get("training", False):
            keep_prob = 1 - self.drop_prob
            shape = (inputs[0],) + (1,) * (inputs.shape[0] - 1)
            random_tensor = keep_prob + keras.random.uniform(shape, 0, 1)
            random_tensor = keras.ops.floor(random_tensor)
            return (inputs / keep_prob) * random_tensor
        return inputs


def cct_mlp(x: keras.KerasTensor, hidden_units: list[int], dropout_rate: float) -> keras.KerasTensor:
    """CCT MPL block

    Args:
        x (keras.KerasTensor): Input tensor
        hidden_units (list[int]): # hidden units
        dropout_rate (float): Dropout rate

    Returns:
        keras.KerasTensor: Output tensor
    """
    for units in hidden_units:
        x = cast(keras.KerasTensor, keras.layers.Dense(units, activation="gelu")(x))
        x = cast(keras.KerasTensor, keras.layers.Dropout(dropout_rate)(x))
    return x


def CCT(
    x: keras.KerasTensor,
    image_size: int,
    num_heads: int,
    projection_dim: int,
    transformer_units: list[int],
    positional_emb: bool,
    num_classes: int,
) -> keras.Model:
    """Generate Compact Convolutional Transformer model (CCT)

    Args:
        x (keras.KerasTensor): Input tensor
        image_size (int): Image size
        num_heads (int): # of heads
        projection_dim (int): Projection dimension
        transformer_units (list[int]): # transformer units
        positional_emb (bool): Enable positional embeddings
        num_classes (int): # classes

    Returns:
        keras.Model: CCT model
    """
    transformer_layers = 2
    stochastic_depth_rate = 0.1

    # Encode patches
    encoded_patches = cct_tokenizer_block()(x)

    # Apply positional embedding.
    if positional_emb:
        dummy_outputs = cct_tokenizer_block()(keras.ops.zeros((1, image_size, image_size, 3)))
        seq_length = dummy_outputs.shape[1]
        projection_dim = dummy_outputs.shape[-1]

        pos_embed = cast(
            Callable,
            keras.layers.Embedding(input_dim=seq_length, output_dim=projection_dim),
        )

        positions = keras.ops.arange(start=0, limit=seq_length, delta=1)
        position_embeddings = cast(keras.KerasTensor, pos_embed(positions))
        encoded_patches += position_embeddings  # type: ignore

    # Calculate Stochastic Depth probabilities.
    dpr = np.linspace(0, stochastic_depth_rate, transformer_layers).tolist()

    # Create multiple layers of the Transformer block.
    for i in range(transformer_layers):
        # Layer normalization 1.
        x1 = keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)

        # Create a multi-head attention layer.
        attention_output = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(
            x1, x1
        )

        # Skip connection 1.
        attention_output = StochasticDepth(dpr[i])(attention_output)
        x2 = keras.layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = cast(keras.KerasTensor, keras.layers.LayerNormalization(epsilon=1e-5)(x2))

        # MLP.
        x3 = cct_mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        # Skip connection 2.
        x3 = StochasticDepth(dpr[i])(x3)
        encoded_patches = keras.layers.Add()([x3, x2])

    # Apply sequence pooling.
    representation = keras.layers.LayerNormalization(epsilon=1e-5)(encoded_patches)
    attention_weights = keras.layers.Dense(1)(representation)
    attention_weights = keras.layers.Softmax(axis=1)(attention_weights)
    weighted_representation = keras.layers.Multiply()([keras.ops.transpose(attention_weights), representation])
    weighted_representation = keras.ops.squeeze(weighted_representation, -2)

    # Classify outputs.
    logits = keras.layers.Dense(num_classes)(weighted_representation)

    # Create the Keras model.
    model = keras.Model(inputs=x, outputs=logits)
    return model
