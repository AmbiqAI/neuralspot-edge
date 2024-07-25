import keras
from .base_augmentation_layer import BaseAugmentationLayer, SAMPLES, TRANSFORMS, LABELS


class RandomAugmentationPipeline(BaseAugmentationLayer):

    def __init__(self,
        layers: list[BaseAugmentationLayer],
        augmentations_per_sample: int = 1,
        rate: float = 1.0,
        seed: int | None = None,
        auto_vectorize: bool = True,
        data_format: str | None = None,
        name: str | None = None,
        **kwargs
    ):
        super().__init__(seed=seed, auto_vectorize=auto_vectorize, data_format=data_format, name=name, **kwargs)
        self.layers = layers
        self.augmentation_per_sample = augmentations_per_sample
        self.rate = rate

    def _random_choice(self, inputs):
        """Randomly choose an augmentation layer."""

        lyr_idx: int = keras.ops.random.randint(
            shape=(), minval=0, maxval=len(self.layers), dtype="int32"
        )
        lyr = self.layers[lyr_idx]

        samples = inputs[SAMPLES]
        labels = inputs.get(LABELS, None)
        result = {}

        batch_size = keras.ops.shape(samples)[0]

        transformations = lyr.get_random_transformations(
            batch_size=batch_size,
            input_shape=keras.ops.shape(samples)
        )

        result[SAMPLES] = lyr.augment_samples(
            inputs={SAMPLES: samples, TRANSFORMS: transformations}
        )

        if labels is not None:
            result[LABELS] = lyr.augment_labels(
                inputs={LABELS: labels, TRANSFORMS: transformations}
            )
        # END IF
        return inputs

    def batch_augment(self, inputs):
        result = dict(inputs)
        for _ in range(self.augmentations_per_sample):
            skip_augment = self._random_generator.uniform(
                shape=(), minval=0.0, maxval=1.0, dtype="float32"
            )
            result = keras.ops.cond(
                skip_augment > self.rate,
                lambda: result,
                lambda: self._random_choice(result),
            )
        # END FOR

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result
