import keras

from .utils import parse_factor
from .base_augmentation_layer import BaseAugmentationLayer, TRANSFORMS, SAMPLES, LABELS


class RandomCutout(keras.Layer):

    factor: tuple[float,float]
    cutouts: tuple[int,int]
    fill_mode: str
    fill_value: float
    seed: int | None
    generator: keras.random.SeedGenerator
    data_format: str

    def __init__(
        self,
        factor: float|tuple[float,float] = 0.1,
        cutouts: int|tuple[int,int] = 1,
        fill_mode="constant",
        fill_value: float = 0.0,
        seed: int | None = None,
        data_format: str | None = None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.factor = parse_factor(factor)
        self.cutouts = parse_factor(cutouts, 1, None, "cutouts")

        self.fill_mode = fill_mode
        self.fill_value = fill_value
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

        if fill_mode not in ["normal", "constant"]:
            raise ValueError(f'`fill_mode` should be "normal" or "constant". Got `fill_mode`={fill_mode}')
        # END IF


    def call(self, inputs, training=True):
        is_batched = inputs.shape.rank == 3
        outputs = inputs
        if not is_batched:
            outputs = keras.ops.expand_dims(outputs, axis=0)
        # END
        input_shape = keras.ops.shape(outputs)

        duration_size = input_shape[self.duration_axis]
        ch_size = input_shape[self.ch_axis]

        def random_cutout(sample):
            cut_size = keras.random.randint(
                shape=(),
                minval=int(duration_size * self.factor[0]),
                maxval=int(duration_size * self.factor[1]),
                dtype="int32",
                seed=self.generator
            )
            cut_start = keras.random.randint(
                shape=(),
                minval=0,
                maxval=duration_size - cut_size,
                dtype="int32",
                seed=self.generator
            )
            if self.fill_mode == "constant":
                fill = keras.ops.ones((cut_size, ch_size)) * self.fill_value
            else:
                fill = keras.random.normal((cut_size, ch_size), mean=0, stddev=self.fill_value, seed=self.generator)
            result = keras.ops.slice_update(sample, start_indices=(cut_start, 0), updates=fill)
            return result

        if training:
            num_cutouts = keras.random.randint(
                shape=(),
                minval=self.cutouts[0],
                maxval=self.cutouts[1] + 1,
                seed=self.generator,
                dtype="int32"
            )
            for _ in range(num_cutouts):
                outputs = keras.ops.map(f=random_cutout, xs=outputs)
            # END FOR
        # END IF

        if not is_batched:
            outputs = keras.ops.squeeze(outputs, axis=0)
        return outputs

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "factor": self.factor,
                "cutouts": self.cutouts,
                "fill_mode": self.fill_mode,
                "fill_value": self.fill_value,
                "seed": self.seed,
                "data_format": self.data_format,
            }
        )
        return config


class RandomCutoutLayer(BaseAugmentationLayer):

    factor: tuple[float,float]
    fill_mode: str
    fill_value: float

    def __init__(
        self,
        factor: float|tuple[float,float] = 0.1,
        fill_mode="constant",
        fill_value: float = 0.0,
        seed: int | None = None,
        auto_vectorize: bool = True,
        data_format: str | None = None,
        name: str | None = None,
        **kwargs
    ):
        super().__init__(seed=seed, auto_vectorize=auto_vectorize, data_format=data_format, name=name, **kwargs)

        self.factor = parse_factor(factor)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        if fill_mode not in ["normal", "constant"]:
            raise ValueError(f'`fill_mode` should be "normal" or "constant". Got `fill_mode`={fill_mode}')
        # END IF

    def augment_sample(self, inputs) -> keras.KerasTensor:
        sample = inputs[SAMPLES]
        transforms = inputs[TRANSFORMS]
        cut_start = transforms["cut_start"]
        cut_size = transforms["cut_size"]
        seed = transforms["seed"]

        ch_size = sample.shape[self.ch_axis]

        if self.fill_mode == "constant":
            fill = keras.ops.ones((cut_size, ch_size)) * self.fill_value
        else:
            fill = keras.random.normal((cut_size, ch_size), mean=0, stddev=self.fill_value, seed=seed)

        return keras.ops.slice_update(sample, start_indices=(cut_start, 0), updates=fill)

    def augment_label(self, inputs) -> keras.KerasTensor:
        return inputs[LABELS]

    def get_random_transformations(self, batch_size: int, input_shape):
        duration_size = input_shape[self.duration_axis]

        cut_size = keras.random.randint(
            shape=(batch_size,),
            minval=int(duration_size * self.factor[0]),
            maxval=int(duration_size * self.factor[1]),
            dtype="int32",
            seed=self._random_generator
        )
        cut_start = keras.ops.map(
            lambda size: keras.random.randint(
                shape=(),
                minval=0,
                maxval=duration_size - size,
                dtype="int32",
                seed=self._random_generator
            ),
            cut_size
        )

        return {
            "cut_start": cut_start,
            "cut_size": cut_size,
            "seed": keras.random.randint(shape=(batch_size,), minval=0, maxval=2**32, seed=self._random_generator, dtype="int32"),
        }
