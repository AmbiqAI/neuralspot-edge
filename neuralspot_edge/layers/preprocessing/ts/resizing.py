import keras


class Resizing(keras.Layer):

    duration: int
    data_format: str

    def __init__(
        self,
        duration: int,
        data_format: str | None = None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.duration = duration

        self.data_format = data_format or keras.backend.image_data_format()

        if self.data_format == "channels_first":
            self.duration_axis = -1
        else:
            self.duration_axis = -2
        # END IF


    def call(self, inputs: keras.KerasTensor, training=True):

        if inputs.shape.rank != 2 and inputs.shape.rank != 3:
            raise ValueError(f"Invalid input shape: {inputs.shape}")

        y = inputs
        is_batched = inputs.shape.rank == 3
        # Add batch dimension if required
        if not is_batched:
            y = keras.ops.expand_dims(y, axis=0)
        # END IF

        # Add height dimension
        y = keras.ops.expand_dims(y, axis=1)
        y = keras.ops.image.resize(
            y,
            size=(1, self.duration),
            interpolation="bicubic",
            crop_to_aspect_ratio=False,
            data_format=self.data_format,
        )
        # Remove height dimension
        y = keras.ops.squeeze(y, axis=1)

        # Remove batch dimension if required
        if not is_batched:
            y = keras.ops.squeeze(y, axis=0)
        return y

    def compute_output_shape(self, input_shape, *args, **kwargs):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            duration=self.duration,
            data_format=self.data_format,
        )
        return config
