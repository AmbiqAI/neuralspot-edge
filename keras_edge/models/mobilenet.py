""" " MobileNet"""

from typing import cast

import keras
from pydantic import BaseModel, Field


class MobileNetV1Params(BaseModel):
    """MobileNetV1 parameters"""

    input_filters: int = Field(default=8, description="Input filters")
    input_strides: int | tuple[int, int] = Field(default=2, description="Input stride")
    include_top: bool = Field(default=True, description="Include top")
    output_activation: str | None = Field(default=None, description="Output activation")
    name: str = Field(default="RegNet", description="Model name")


def MobileNetV1(  # pylint: disable=too-many-statements
    x: keras.KerasTensor,
    num_classes: int,
    params: MobileNetV1Params,
) -> keras.Model:
    """Modified MobileNetV1
    MLPerf Tiny model for VWW:
    https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/mobilenet_v1_eembc.py

    Args:
        x (keras.KerasTensor): Model input
        num_classes (int): # classes.
        params (MobileNetV1Params): Model parameters

    Returns:
        keras.Model: Model
    """
    num_filters = params.input_filters

    # 1st layer, pure conv
    # Keras 2.2 model has padding='valid' and disables bias
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=3,
        strides=params.input_strides,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(x)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)  # Keras uses ReLU6 instead of pure ReLU

    # 2nd layer, depthwise separable conv
    # Filter size is always doubled before the pointwise conv
    # Keras uses ZeroPadding2D() and padding='valid'
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 3rd layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 4th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 5th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 6th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 7th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 8th-12th layers, identical depthwise separable convs
    for _ in range(8, 13):
        y = keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

        y = keras.layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=1,
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=keras.regularizers.l2(1e-4),
        )(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation("relu")(y)

    # 13th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=2,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    num_filters = 2 * num_filters
    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    # 14th layer, depthwise separable conv
    y = keras.layers.DepthwiseConv2D(
        kernel_size=3,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = keras.layers.Activation("relu")(y)

    y = keras.layers.Conv2D(
        num_filters,
        kernel_size=1,
        strides=1,
        padding="same",
        kernel_initializer="he_normal",
        kernel_regularizer=keras.regularizers.l2(1e-4),
    )(y)
    y = keras.layers.BatchNormalization()(y)
    y = cast(keras.KerasTensor, keras.layers.Activation("relu")(y))

    # Average pooling, max polling may be used also
    # Keras employs GlobalAveragePooling2D
    y = keras.layers.AveragePooling2D(pool_size=y.shape[1:3])(y)
    # x = MaxPooling2D(pool_size=x.shape[1:3])(x)

    # Keras inserts Dropout() and a pointwise Conv2D() here
    # We are staying with the paper base structure

    # Flatten, FC layer and classify
    if params.include_top:
        y = keras.layers.Flatten()(y)
        y = keras.layers.Dense(num_classes)(y)
    model = keras.Model(x, y, name="model")
    return model
