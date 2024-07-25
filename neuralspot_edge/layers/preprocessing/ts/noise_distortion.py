import keras


class NoiseDistortion(keras.Layer):
    def __init__(
        self,
        sample_rate: float = 1,
        frequency: float = 100,
        amplitude: float = 0.1,
        noise_type: str = "normal",
        seed: int | None = None,
        data_format: str | None = None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.amplitude = amplitude
        self.noise_type = noise_type
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


    def call(self, inputs: keras.KerasTensor, training: bool = True):

        if inputs.shape.rank != 2 and inputs.shape.rank != 3:
            raise ValueError(f"Invalid input shape: {inputs.shape}")

        # Force into batched mode (B, T, C)
        is_batched = inputs.shape.rank == 3
        if not is_batched:
            y = keras.ops.expand_dims(inputs, axis=0)
        else:
            y = inputs

        if training:
            input_shape = keras.ops.shape(y)
            batch_size = input_shape[0]
            duration_size = input_shape[self.duration_axis]
            ch_size = input_shape[self.ch_axis]

            # Add one period to the noise and clip later
            noise_duration = int(duration_size * self.frequency + self.frequency)

            if self.data_format == "channels_first":
                noise_shape = (batch_size, 1, ch_size, noise_duration)
            else:
                noise_shape = (batch_size, 1, noise_duration, ch_size)

            if self.noise_type == "normal":
                noise_pts = keras.random.normal(noise_shape, stddev=self.amplitude, seed=self.generator)
            else:
                raise ValueError(f"Invalid noise shape: {self.noise_type}")

            # keras.ops doesnt contain any low-level interpolate. So we leverage the
            # image module and fix height to 1 as workaround
            noise = keras.ops.image.resize(
                noise_pts,
                size=(1, duration_size),
                interpolation="bicubic",
                crop_to_aspect_ratio=False,
                data_format=self.data_format,
            )
            # Remove height dimension
            noise = keras.ops.squeeze(noise, axis=1)

            y = keras.ops.cast(y + noise, self.compute_dtype)
        # END IF

        # Remove batch dimension if not batched
        if not is_batched:
            y = keras.ops.squeeze(y, axis=0)
        return y

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sample_rate": self.sample_rate,
                "frequency": self.frequency,
                "amplitude": self.amplitude,
                "noise_shape": self.noise_type,
                "seed": self.seed,
                "data_format": self.data_format,
            }
        )
        return config
