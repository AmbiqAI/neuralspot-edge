import keras

SAMPLES = "data"
LABELS = "labels"
TARGETS = "targets"
ALL_KEYS = (SAMPLES, LABELS, TARGETS)
TRANSFORMS = "transforms"
IS_DICT = "is_dict"
BATCHED = "is_batched"
USE_TARGETS = "use_targets"


class BaseAugmentationLayer(keras.layers.Layer):

    def __init__(
        self,
        seed: int | None = None,
        auto_vectorize: bool = True,
        data_format: str | None = None,
        name: str | None = None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._random_generator = keras.random.SeedGenerator(seed)
        self.data_format = data_format or keras.backend.image_data_format()
        self.built = True
        self.training = True
        self.auto_vectorize = auto_vectorize

        if self.data_format == "channels_first":
            self.duration_axis = -1
            self.ch_axis = -2
        else:
            self.duration_axis = -2
            self.ch_axis = -1
        # END IF

    def _map_fn(self, func, inputs):
        """Vectorized map function."""
        if self.auto_vectorize:
            return keras.ops.vectorized_map(func, inputs)
        raise NotImplementedError("Non-vectorized map function is not implemented.")

    def call(self, inputs, training: bool = True):
        self.training = training
        inputs, metadata = self._format_inputs(inputs)
        return self._format_outputs(self.batch_augment(inputs), metadata)

    def augment_sample(self, inputs) -> keras.KerasTensor:
        """Augment a single sample during training.

        Args:
            input(dict): A dictionary containing samples and transformations.

        Returns:
            KerasTensor: Augmented sample.
        """
        raise NotImplementedError()

    def augment_samples(self, inputs) -> keras.KerasTensor:
        """Augment a batch of samples during training.

        Args:
            input(dict): A dictionary containing samples and transformations.

        Returns:
            KerasTensor: Augmented samples.
        """
        return self._map_fn(self.augment_sample, inputs=inputs)

    def augment_label(self, inputs) -> keras.KerasTensor:
        """Augment a single label during training.

        Args:
            input(dict): A dictionary containing labels and transformations.

        Returns:
            keras.KerasTensor: Augmented label.
        """
        raise NotImplementedError()

    def augment_labels(self, inputs) -> keras.KerasTensor:
        """Augment a batch of labels during training.

        Args:
            input(dict): A dictionary containing labels and transformations.

        Returns:
            keras.KerasTensor: Augmented labels.
        """
        return self._map_fn(self.augment_label, inputs=inputs)

    def get_random_transformations(
        self,
        batch_size: int,
        input_shape
    ):
        return keras.ops.arange(batch_size)

    def batch_augment(self, inputs):
        samples = inputs.get(SAMPLES, None)
        labels = inputs.get(LABELS, None)
        result = {}
        batch_size = keras.ops.shape(samples)[0]

        transformations = self.get_random_transformations(
            batch_size=batch_size,
            input_shape=keras.ops.shape(samples)
        )

        result[SAMPLES] = self.augment_samples(
            inputs={SAMPLES: samples, TRANSFORMS: transformations}
        )

        if labels is not None:
            result[LABELS] = self.augment_labels(
                inputs={LABELS: labels, TRANSFORMS: transformations}
            )
        # END IF

        # preserve any additional inputs unmodified by this layer.
        for key in inputs.keys() - result.keys():
            result[key] = inputs[key]
        return result

    def _format_inputs(self, inputs):
        metadata = {IS_DICT: True, USE_TARGETS: False, BATCHED: True}
        if not isinstance(inputs, dict):
            inputs = {SAMPLES: inputs}
            metadata[IS_DICT] = False

        samples = inputs.get(SAMPLES, None)
        if inputs.get(SAMPLES) is None:
            raise ValueError(f"Expect the inputs to have key {SAMPLES}. Got keys: {list(inputs.keys())}")
        # END IF
        if inputs[SAMPLES].shape.rank != 2 and samples.shape.rank != 3:
            raise ValueError(f"Invalid input shape: {samples.shape}")
        # END IF
        if inputs[SAMPLES].shape.rank == 2:
            metadata[BATCHED] = False
            # Expand dims to make it batched for keys of interest
            for key in set(ALL_KEYS).intersection(inputs.keys()):
                if inputs[key] is not None:
                    inputs[key] = keras.ops.expand_dims(inputs[key], axis=0)
                # END IF
            # END FOR
        # END IF
        return inputs, metadata

    def _format_outputs(self, output, metadata):
        if not metadata[BATCHED]:
            for key in set(ALL_KEYS).intersection(output.keys()):
                if output[key] is not None: # check if tensor
                    output[key] = keras.ops.squeeze(output[key], axis=0)
                # END IF
            # END FOR
        # END IF
        if not metadata[IS_DICT]:
            return output[SAMPLES]
        if metadata[USE_TARGETS]:
            output[TARGETS] = output[LABELS]
            del output[LABELS]
        return output



BaseAugmentationLayer.SAMPLERS
