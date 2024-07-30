import keras
from .base_augmentation import BaseAugmentation1D


class RandomAugmentation1DPipeline(BaseAugmentation1D):
    layers: list[BaseAugmentation1D]
    augmentations_per_sample: int
    rate: float

    def __init__(
        self, layers: list[BaseAugmentation1D], augmentations_per_sample: int = 1, rate: float = 1.0, **kwargs
    ):
        """Apply N random augmentations from a list of augmentation layers to each sample.

        Args:
            layers (list[BaseAugmentation1D]): List of augmentation layers to choose from.
            augmentations_per_sample (int): Number of augmentations to apply to each sample.
            rate (float): Probability of applying the augmentation pipeline.
        """
        super().__init__(**kwargs)
        self.layers = layers
        self.augmentations_per_sample = augmentations_per_sample
        self.rate = rate

    def _random_choice(self, inputs):
        """Randomly choose an augmentation layer."""

        lyr_idx: int = keras.random.randint(shape=(), minval=0, maxval=len(self.layers), dtype="int32")
        lyr = self.layers[lyr_idx]
        return lyr.batch_augment(inputs)

    def batch_augment(self, inputs):
        """Apply N random augmentations to each"""
        result = dict(inputs)
        for _ in range(self.augmentations_per_sample):
            skip_augment = keras.random.uniform(
                shape=(), minval=0.0, maxval=1.0, dtype="float32", seed=self._random_generator
            )
            result = keras.ops.cond(
                skip_augment > self.rate,
                lambda: result,
                lambda: self._random_choice(result),
            )
        # END FOR
        return result

    def get_config(self):
        """Serializes the configuration of the layer."""
        config = super().get_config()
        config.update(
            {
                "layers": [lyr.get_config() for lyr in self.layers],
                "augmentations_per_sample": self.augmentations_per_sample,
                "rate": self.rate,
            }
        )
        return config
