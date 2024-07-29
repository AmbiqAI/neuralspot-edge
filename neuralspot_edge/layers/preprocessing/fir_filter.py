import numpy as np
import numpy.typing as npt
import keras

from .base_augmentation import BaseAugmentation1D


class FirFilter(BaseAugmentation1D):
    def __init__(
        self,
        b: npt.NDArray[np.float32],
        a: npt.NDArray[np.float32],
        forward_backward: bool = False,
        seed: int | None = None,
        data_format: str | None = None,
        auto_vectorize: bool = True,
        name=None,
        **kwargs,
    ):
        """Apply FIR filter to the input.

        Args:
            b (np.ndarray): Numerator coefficients.
            a (np.ndarray): Denominator coefficients.
            forward_backward (bool): Apply filter forward and backward.
        """
        super().__init__(seed=seed, data_format=data_format, auto_vectorize=auto_vectorize, name=name, **kwargs)
        b = b.reshape(-1, 1, 1)
        a = a.reshape(-1, 1, 1)
        self.b = self.add_weight(
            name="b",
            shape=b.shape,
            trainable=False,
        )
        self.b.assign(b)
        self.a = self.add_weight(
            name="a",
            shape=a.shape,
            trainable=False,
        )
        self.a.assign(a)

        self.forward_backward = forward_backward

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment a batch of samples during training."""

        samples = inputs[self.SAMPLES]

        # TODO: Handle 'a' coefficients
        if self.a is not None:
            raise NotImplementedError("Denominator coefficients 'a' are not supported yet.")

        outputs = keras.ops.depthwise_conv(samples, self.b, padding="same")

        if self.forward_backward:
            outputs = keras.ops.flip(outputs, axis=self.data_axis)
            outputs = keras.ops.depthwise_conv(outputs, self.b, padding="same")
            outputs = keras.ops.flip(outputs, axis=self.data_axis)
        # END IF
        return outputs

    def get_config(self):
        """Serialize the layer configuration."""
        config = super().get_config()
        config.update(
            {
                "forward_backward": self.forward_backward,
            }
        )
        return config
