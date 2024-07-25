import keras
from .base_augmentation_layer import BaseAugmentationLayer, SAMPLES, TRANSFORMS

class RandomCrop(keras.Layer):

    duration: int
    unique_batch: bool
    seed: int | None
    generator: keras.random.SeedGenerator
    data_format: str

    def __init__(
            self,
            duration: int,
            unique_batch: bool = False,
            seed: int | None = None,
            data_format: str | None = None,
            name=None,
            **kwargs
        ):
        """Randomly crops a temporal sequence to a fixed length

        Similar to keras.layers.RandomCrop but for 1D sequences.

        Args:
            duration (int): Duration of the output sequence.
            unique_batch (bool): If True, each sample in the batch is cropped independently.
            seed (int | None): Random seed.
            data_format (str | None): Data format. Defaults to None.
            name (str | None): Layer name. Defaults to None.
        """

        super().__init__(name=name, **kwargs)
        self.duration = duration
        self.unique_batch = unique_batch
        self.seed = seed
        self.generator = keras.random.SeedGenerator(seed)
        self.data_format = data_format or keras.backend.image_data_format()

        if self.data_format == "channels_first":
            self.duration_axis = -1
            self.ch_axis = -2
        else:
            self.duration_axis = -2
            self.ch_axis = -1
        # END IF

    def call(self, inputs, training=True):
        inputs = keras.ops.cast(inputs, self.compute_dtype)

        is_batched = inputs.shape.rank == 3
        if not is_batched:
            inputs = keras.ops.expand_dims(inputs, axis=0)

        input_shape = keras.ops.shape(inputs)
        input_size = input_shape[self.duration_axis]
        ch_size = input_shape[self.ch_axis]
        d_diff = input_size - self.duration

        def random_crop(sample):
            """Randomly crop single sample"""
            start = keras.random.randint(shape=(), minval=0, maxval=int(d_diff + 1), seed=self.generator, dtype="int32")
            if self.data_format == "channels_first":
                return keras.ops.slice(sample, [0, start], [ch_size, self.duration])
            return keras.ops.slice(sample, [start, 0], [self.duration, ch_size])
            # END IF
        # END DEF

        def resize(sample):
            """Randomly resize single sample"""
            if self.data_format == "channels_first":
                sample = keras.ops.pad(sample, [[0, 0], [0, d_diff]])
            else:
                sample = keras.ops.pad(sample, [[0, d_diff], [0, 0]])
            return keras.ops.cast(sample, self.compute_dtype)
        # END DEF
        map_fn = keras.ops.map if self.unique_batch else keras.ops.vectorized_map

        if training and d_diff >= 0:
            outputs = map_fn(random_crop, inputs)
        elif d_diff < 0:
            outputs = map_fn(resize, inputs)
        else:
            outputs = inputs
        if not is_batched:
            outputs = keras.ops.squeeze(outputs, axis=0)
        return outputs

    def compute_output_shape(self, input_shape, *args, **kwargs):
        input_shape = list(input_shape)
        input_shape[self.duration_axis] = self.duration
        return tuple(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "duration": self.duration,
                "unique_batch": self.unique_batch,
                "seed": self.seed,
                "data_format": self.data_format,
            }
        )
        return config


class RandomCropLayer(BaseAugmentationLayer):

    def __init__(
            self,
            duration: int,
            unique_batch: bool = False,
            seed: int | None = None,
            data_format: str | None = None,
            auto_vectorize: bool = True,
            name=None,
            **kwargs
        ):

        super().__init__(
            seed=seed,
            data_format=data_format,
            auto_vectorize=auto_vectorize,
            name=name,
            **kwargs
        )
        self.duration = duration
        self.unique_batch = unique_batch

    def random_crop(self, sample, start):
        """Randomly crop single sample"""
        ch_size = keras.ops.shape(sample)[self.ch_axis]
        if self.data_format == "channels_first":
            return keras.ops.slice(sample, [0, start], [ch_size, self.duration])
        return keras.ops.slice(sample, [start, 0], [self.duration, ch_size])
        # END IF
    # END DEF

    def resize(self, sample):
        """Randomly resize single sample"""
        input_size = keras.ops.shape(sample)[self.duration_axis]
        d_diff = self.duration - input_size
        if d_diff < 0:
            return sample
        if self.data_format == "channels_first":
            sample = keras.ops.pad(sample, [[0, 0], [0, d_diff]])
        else:
            sample = keras.ops.pad(sample, [[0, d_diff], [0, 0]])
        return keras.ops.cast(sample, self.compute_dtype)
    # END DEF

    def get_random_transformations(
        self,
        batch_size: int,
        input_shape
    ):
        # Return random start indices for cropping
        d_diff = input_shape[self.duration_axis] - self.duration
        if d_diff <= 0:
            return d_diff*keras.ops.ones((batch_size,), dtype="int32")
        starts = keras.random.randint(
            shape=(batch_size,),
            minval=0,
            maxval=int(d_diff + 1),
            seed=self._random_generator,
            dtype="int32"
        )
        return starts

    def augment_sample(self, inputs) -> dict:
        sample = inputs[SAMPLES]
        start = inputs[TRANSFORMS]

        sample = keras.ops.cond(
            keras.ops.logical_and(self.training, start >= 0),
            lambda: self.random_crop(sample, start),
            lambda: self.resize(sample),
        )
        return sample

    def augment_label(self, inputs) -> dict:
        return inputs['LABELS']

    def compute_output_shape(self, input_shape, *args, **kwargs):
        input_shape = list(input_shape)
        input_shape[self.duration_axis] = self.duration
        return tuple(input_shape)
