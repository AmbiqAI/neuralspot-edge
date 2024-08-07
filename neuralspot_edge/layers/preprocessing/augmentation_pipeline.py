import keras


class AugmentationPipeline(keras.Layer):
    def __init__(
        self,
        layers: list[keras.Layer],
        name: str | None = None,
        force_training: bool = False,
    ):
        """Pipeline of augmentation layers.

        Args:
            layers (list[keras.Layer]): List of augmentation layers.
            force_training (bool, optional): Force training mode. Defaults to False.
        """
        super().__init__(name=name)
        self.layers = layers
        self.force_training = force_training

    def call(self, inputs, training: bool = True, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, training=training or self.force_training, **kwargs)
        return inputs
