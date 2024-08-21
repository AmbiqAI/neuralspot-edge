import keras
from .base_augmentation import BaseAugmentation1D, BaseAugmentation2D
from ...utils import nse_export


@nse_export(path="neuralspot_edge.layers.preprocessing.Rescaling1D")
class Rescaling1D(BaseAugmentation1D):
    scale: float

    def __init__(self, scale: float, **kwargs):
        """Rescale the input samples.

        Args:
            scale (float): The scaling factor.
        """
        super().__init__(**kwargs)
        self.scale = scale

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Rescale a batch of samples during training."""
        samples = inputs[self.SAMPLES]
        return samples * self.scale

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(scale=self.scale)
        return config


@nse_export(path="neuralspot_edge.layers.preprocessing.Rescaling2D")
class Rescaling2D(BaseAugmentation2D):
    scale: float

    def __init__(self, scale: float, **kwargs):
        """Rescale the input samples.

        Args:
            scale (float): The scaling factor.
        """
        super().__init__(**kwargs)
        self.scale = scale

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Rescale a batch of samples during training."""
        samples = inputs[self.SAMPLES]
        return samples * self.scale

    def compute_output_shape(self, input_shape, *args, **kwargs):
        """Compute output shape."""
        return input_shape

    def get_config(self):
        """Serialize the configuration."""
        config = super().get_config()
        config.update(scale=self.scale)
        return config
