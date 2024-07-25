import keras

from .utils import parse_factor

class RandomBackgroundNoises(keras.Layer):

    amplitude: tuple[float,float]
    num_noises: tuple[int,int]

    def __init__(
        self,
        noises,
        amplitude: float|tuple[float,float] = 0.1,
        num_noises: int|tuple[int,int] = 1,
        seed: int | None = None,
        data_format: str | None = None,
        name=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.amplitude = parse_factor(amplitude, min_value=0, max_value=None, param_name="amplitude")
        self.num_noises = parse_factor(num_noises, min_value=1, max_value=None, param_name="num_noises")

        self.seed = seed
        self.generator = keras.random.SeedGenerator(seed)
        self.data_format = data_format or keras.backend.image_data_format()
        self.noises = self.add_weight(
            name="noises",
            shape=noises.shape,
            trainable=False,
        )
        self.noises.assign(noises)

        if self.data_format == "channels_first":
            self.duration_axis = -1
            self.ch_axis = -2
        else:
            self.duration_axis = -2
            self.ch_axis = -1
        # END IF


    def call(self, inputs, training=True):
        is_batched = inputs.shape.rank == 3
        if not is_batched:
            inputs = keras.ops.expand_dims(inputs, axis=0)
        # END

        input_shape = keras.ops.shape(inputs)
        batch_size = input_shape[0]
        duration_size = input_shape[self.duration_axis]
        ch_size = input_shape[self.ch_axis]

        # noises have shape (T, nnoises) assume T is longer than the input
        noise_shape = keras.ops.shape(self.noises)
        noise_size = noise_shape[0]
        num_background_noises = noise_shape[1]

        num_noises = keras.random.randint(shape=(), minval=self.num_noises[0], maxval=self.num_noises[1], seed=self.generator, dtype="int32")
        amplitude = keras.random.uniform(shape=(), minval=self.amplitude[0], maxval=self.amplitude[1], seed=self.generator, dtype=self.compute_dtype)/keras.ops.cast(num_noises, self.compute_dtype)

        def random_noise(sample):
            """Randomly add noise to single sample"""
            noise_idx = keras.random.randint(shape=(), minval=0, maxval=num_background_noises, seed=self.generator)
            start = keras.random.randint(shape=(), minval=0, maxval=noise_size - duration_size + 1, seed=self.generator, dtype="int32")
            noise = keras.ops.slice(self.noises, (start, noise_idx), (duration_size, 1))
            noise = keras.ops.squeeze(noise)
            if self.data_format == "channels_first":
                noise = keras.ops.reshape(noise, (1, duration_size))
                noise = keras.ops.tile(noise, (ch_size, 1))
            else:
                noise = keras.ops.reshape(noise, (duration_size, 1))
                noise = keras.ops.tile(noise, (1, ch_size))
            return sample + amplitude * noise
        # END DEF


        if training:
            for _ in range(num_noises):
                outputs = keras.ops.map(random_noise, inputs)
        else:
            outputs = inputs

        if not is_batched:
            outputs = keras.ops.squeeze(outputs, axis=0)
        return outputs

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "amplitude": self.amplitude,
                "num_noises": self.num_noises,
                "seed": self.seed,
                "data_format": self.data_format,
            }
        )
        return config
