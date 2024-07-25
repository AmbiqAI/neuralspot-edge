import keras


class GaussianNoise(keras.Layer):
    def __init__(self, stddev: float, seed: int | None = None, name=None, **kwargs):
        """Apply additive zero-centered Gaussian noise.

        Args:
            stddev(float): Standard deviation of the Gaussian noise.
            seed(int | None): Random seed.
            name(str | None): Layer name. Defaults to None.

        Example:
            >>> x = np.sin(2*np.pi*10*np.arange(duration_size)/100)
            >>> lyr = GaussianNoise(stddev=0.1)
            >>> y = lyr(x)
        """
        super().__init__(name=name, **kwargs)
        if not 0 <= stddev <= 1:
            raise ValueError(
                f"Invalid value received for argument "
                "`stddev`. Expected a float value between 0 and 1. "
                f"Received: stddev={stddev}"
            )
        self.stddev = stddev
        self.seed = seed
        self.generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=True):
        """Apply Gaussian noise to the inputs."""
        if training and self.stddev > 0:
            return inputs + keras.random.normal(
                shape=keras.ops.shape(inputs),
                stddev=self.stddev,
                dtype=self.compute_dtype,
                seed=self.generator
            )
        return inputs

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stddev": self.stddev,
                "seed": self.seed
            }
        )
        return config
