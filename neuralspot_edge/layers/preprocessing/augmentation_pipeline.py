import keras


class AugmentationPipeline(keras.Layer):
    def __init__(
        self,
        layers: list[keras.Layer],
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.layers = layers

    def call(self, inputs, **kwargs):
        for layer in self.layers:
            inputs = layer(inputs, **kwargs)
        return inputs
