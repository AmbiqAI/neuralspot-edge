from abc import abstractmethod
from typing import Callable

import keras
import tensorflow as tf


class ContrastiveModel(keras.Model):
    """Base class for contrastive learning models"""

    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model,
        contrastive_augmenter: Callable[[keras.KerasTensor], keras.KerasTensor] | None = None,
        classification_augmenter: Callable[[keras.KerasTensor], keras.KerasTensor] | None = None,
        linear_probe: keras.Model | None = None,
    ):
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.contrastive_augmenter = contrastive_augmenter
        self.classification_augmenter = classification_augmenter
        self.linear_probe = linear_probe

        self.probe_loss = None
        self.probe_optimizer = None
        self.contrastive_loss_tracker = None
        self.contrastive_optimizer = None
        self.contrastive_accuracy = None
        self.correlation_accuracy = None
        self.probe_accuracy = None

    @property
    def metrics(self):
        """List of metrics to track during training and evaluation"""
        return [
            self.contrastive_loss_tracker,
            self.correlation_accuracy,
            self.contrastive_accuracy,
            # self.probe_loss_tracker,
            # self.probe_accuracy,
        ]

    @abstractmethod
    def contrastive_loss(self, projections_1, projections_2):
        """Contrastive loss function"""
        raise NotImplementedError()

    def call(self, inputs, training=None, mask=None):
        """Forward pass through the encoder model"""
        return self.encoder(inputs, training=training, mask=mask)

    # pylint: disable=unused-argument,arguments-differ
    def compile(
        self,
        contrastive_optimizer: keras.optimizers.Optimizer,
        probe_optimizer: keras.optimizers.Optimizer | None = None,
        **kwargs,
    ):
        """Compile the model with the specified optimizers"""
        super().compile(**kwargs)

        self.contrastive_optimizer = contrastive_optimizer
        self.probe_optimizer = probe_optimizer

        # self.contrastive_loss is a method that will be implemented by the subclasses
        self.probe_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.contrastive_loss_tracker = keras.metrics.Mean(name="loss")
        self.contrastive_accuracy = keras.metrics.SparseCategoricalAccuracy(name="c_acc")
        self.correlation_accuracy = keras.metrics.SparseCategoricalAccuracy(name="r_acc")

        self.probe_accuracy = keras.metrics.SparseCategoricalAccuracy()

    def save(self, filepath, overwrite=True, save_format=None, **kwargs):
        """Save the encoder model to file

        Args:
            filepath (str): Filepath
            overwrite (bool, optional): Overwrite existing file. Defaults to True.
            save_format ([type], optional): Save format. Defaults to None.
        """
        self.encoder.save(filepath, overwrite, save_format, **kwargs)

    def reset_metrics(self):
        """Reset the metrics to their initial state"""
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()
        self.probe_accuracy.reset_states()

    def update_contrastive_accuracy(self, features_1, features_2):
        """Update the contrastive accuracy metric
        self-supervised metric inspired by the SimCLR loss
        """

        # cosine similarity: the dot product of the l2-normalized feature vectors
        features_1 = keras.ops.normalize(features_1, axis=1)
        features_2 = keras.ops.normalize(features_2, axis=1)
        similarities = keras.ops.matmul(features_1, keras.ops.transpose(features_2))

        # Push positive pairs to the diagonal
        batch_size = keras.ops.shape(features_1)[0]
        contrastive_labels = keras.ops.arange(batch_size)
        self.contrastive_accuracy.update_state(contrastive_labels, similarities)
        self.contrastive_accuracy.update_state(contrastive_labels, keras.ops.transpose(similarities))

    def update_correlation_accuracy(self, features_1, features_2):
        """Update the correlation accuracy metric
        self-supervised metric inspired by the BarlowTwins loss
        """

        # normalization so that cross-correlation will be between -1 and 1
        features_1 = (features_1 - keras.ops.mean(features_1, axis=0)) / keras.ops.std(features_1, axis=0)
        features_2 = (features_2 - keras.ops.mean(features_2, axis=0)) / keras.ops.std(features_2, axis=0)

        # the cross correlation of image representations should be the identity matrix
        batch_size = keras.ops.shape(features_1)[0]
        batch_size = keras.ops.cast(batch_size, dtype="float32")
        cross_correlation = keras.ops.matmul(features_1, keras.ops.transpose(features_2)) / batch_size

        feature_dim = keras.ops.shape(features_1)[1]
        correlation_labels = keras.ops.arange(feature_dim)
        self.correlation_accuracy.update_state(correlation_labels, cross_correlation)
        self.correlation_accuracy.update_state(correlation_labels, keras.ops.transpose(cross_correlation))

    def train_step(self, data):
        """Training step for the model"""
        pair1, pair2 = data

        # each input is augmented twice, differently
        augmented_inputs_1 = self.contrastive_augmenter(pair1)
        augmented_inputs_2 = self.contrastive_augmenter(pair2)
        with tf.GradientTape() as tape:
            # Encoder phase
            features_1 = self.encoder(augmented_inputs_1)
            features_2 = self.encoder(augmented_inputs_2)
            # Projection phase
            projections_1 = self.projector(features_1)
            projections_2 = self.projector(features_2)
            contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        # END WITH

        # backpropagation
        gradients = tape.gradient(
            contrastive_loss,
            self.encoder.trainable_weights + self.projector.trainable_weights,
        )
        self.contrastive_optimizer.apply_gradients(
            zip(
                gradients,
                self.encoder.trainable_weights + self.projector.trainable_weights,
            )
        )

        self.contrastive_loss_tracker.update_state(contrastive_loss)

        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        # # labels are only used in evalutation for probing
        # augmented_inputs = self.classification_augmenter(labeled_pair)
        # with tf.GradientTape() as tape:
        #     features = self.encoder(augmented_inputs)
        #     class_logits = self.linear_probe(features)
        #     probe_loss = self.probe_loss(labels, class_logits)
        # gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)
        # self.probe_optimizer.apply_gradients(
        #     zip(gradients, self.linear_probe.trainable_weights)
        # )
        # self.probe_accuracy.update_state(labels, class_logits)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """Test step for the model"""
        pair1, pair2 = data
        augmented_inputs_1 = self.contrastive_augmenter(pair1)
        augmented_inputs_2 = self.contrastive_augmenter(pair2)
        features_1 = self.encoder(augmented_inputs_1, training=False)
        features_2 = self.encoder(augmented_inputs_2, training=False)
        projections_1 = self.projector(features_1, training=False)
        projections_2 = self.projector(features_2, training=False)

        contrastive_loss = self.contrastive_loss(projections_1, projections_2)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        self.update_contrastive_accuracy(features_1, features_2)
        self.update_correlation_accuracy(features_1, features_2)

        return {m.name: m.result() for m in self.metrics}
