import keras
from .base_augmentation import BaseAugmentation2D
from ...utils import nse_export


@nse_export(path="neuralspot_edge.layers.preprocessing.RandomFlip2D")
class RandomFlip2D(BaseAugmentation2D):
    """A preprocessing layer which randomly flips images during training.

    This layer will flip the images horizontally and or vertically based on the
    `mode` attribute. During inference time, the output will be identical to
    input. Call the layer with `training=True` to flip the input.
    Input pixel values can be of any range (e.g. `[0., 1.)` or `[0, 255]`) and
    of integer or floating point dtype.
    By default, the layer will output floats.

    **Note:** This layer is safe to use inside a `tf.data` pipeline
    (independently of which backend you're using).

    Input shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Output shape:
        3D (unbatched) or 4D (batched) tensor with shape:
        `(..., height, width, channels)`, in `"channels_last"` format.

    Args:
        mode: String indicating which flip mode to use. Can be `"horizontal"`,
            `"vertical"`, or `"horizontal_and_vertical"`. `"horizontal"` is a
            left-right flip and `"vertical"` is a top-bottom flip. Defaults to
            `"horizontal_and_vertical"`
        seed: Integer. Used to create a random seed.
        **kwargs: Base layer keyword arguments, such as
            `name` and `dtype`.
    """

    horizontal: bool
    vertical: bool

    def __init__(self, horizontal: bool = True, vertical: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.horizontal = horizontal
        self.vertical = vertical

    def get_random_transformations(self, input_shape):
        """Generate random flip transformations."""
        transforms = {}
        batch_size = input_shape[0]

        if self.horizontal:
            horizontal = self.backend.random.uniform(shape=(batch_size, 1, 1, 1), seed=self.random_generator)
            transforms["horizontal"] = horizontal
        if self.vertical:
            vertical = self.backend.random.uniform(shape=(batch_size, 1, 1, 1), seed=self.random_generator)
            transforms["vertical"] = vertical
        return transforms

    def augment_samples(self, inputs):
        data = inputs[self.SAMPLES]
        if self.horizontal:
            data = self.backend.numpy.where(
                inputs[self.TRANSFORMS]["horizontal"] <= 0.5,
                self.backend.numpy.flip(data, axis=self.height_axis),
                data,
            )
        if self.vertical:
            data = self.backend.numpy.where(
                inputs[self.TRANSFORMS]["vertical"] <= 0.5,
                self.backend.numpy.flip(data, axis=self.width_axis),
                data,
            )

        return data

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "horizontal": self.horizontal,
                "vertical": self.vertical,
            }
        )
        return config
