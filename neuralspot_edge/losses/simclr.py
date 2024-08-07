import keras

from ..utils import nse_export

LARGE_NUM = 1e9


def l2_normalize(x, axis):
    epsilon = keras.backend.epsilon()
    power_sum = keras.ops.sum(keras.ops.square(x), axis=axis, keepdims=True)
    norm = keras.ops.reciprocal(keras.ops.sqrt(keras.ops.maximum(power_sum, epsilon)))
    return keras.ops.multiply(x, norm)


@nse_export(path="neuralspot_edge.losses.SimCLRLoss")
class SimCLRLoss(keras.losses.Loss):
    """Implements SimCLR Cosine Similarity loss.

    SimCLR loss is used for contrastive self-supervised learning.

    Args:
        temperature: a float value between 0 and 1, used as a scaling factor for
            cosine similarity.

    References:
        - [SimCLR paper](https://arxiv.org/pdf/2002.05709)
    """

    def __init__(self, temperature, **kwargs):
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, projections_1, projections_2):
        """Computes SimCLR loss for a pair of projections in a contrastive
        learning trainer.

        Note that unlike most loss functions, this should not be called with
        y_true and y_pred, but with two unlabeled projections. It can otherwise
        be treated as a normal loss function.

        Args:
            projections_1: a tensor with the output of the first projection
                model in a contrastive learning trainer
            projections_2: a tensor with the output of the second projection
                model in a contrastive learning trainer

        Returns:
            A tensor with the SimCLR loss computed from the input projections
        """
        # Normalize the projections
        projections_1 = l2_normalize(projections_1, axis=1)
        projections_2 = l2_normalize(projections_2, axis=1)

        # Produce artificial labels, 1 for each image in the batch.
        batch_size = keras.ops.shape(projections_1)[0]
        labels = keras.ops.one_hot(keras.ops.arange(batch_size), batch_size * 2)
        masks = keras.ops.one_hot(keras.ops.arange(batch_size), batch_size)

        # Compute logits
        logits_11 = keras.ops.matmul(projections_1, keras.ops.transpose(projections_1)) / self.temperature
        logits_11 = logits_11 - keras.ops.cast(masks * LARGE_NUM, logits_11.dtype)
        logits_22 = keras.ops.matmul(projections_2, keras.ops.transpose(projections_2)) / self.temperature
        logits_22 = logits_22 - keras.ops.cast(masks * LARGE_NUM, logits_22.dtype)
        logits_12 = keras.ops.matmul(projections_1, keras.ops.transpose(projections_2)) / self.temperature
        logits_21 = keras.ops.matmul(projections_2, keras.ops.transpose(projections_1)) / self.temperature

        loss_a = keras.losses.categorical_crossentropy(
            labels, keras.ops.concatenate([logits_12, logits_11], 1), from_logits=True
        )
        loss_b = keras.losses.categorical_crossentropy(
            labels, keras.ops.concatenate([logits_21, logits_22], 1), from_logits=True
        )

        return loss_a + loss_b

    def get_config(self):
        config = super().get_config()
        config.update({"temperature": self.temperature})
        return config
