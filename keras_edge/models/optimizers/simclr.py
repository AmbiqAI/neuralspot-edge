from typing import Callable

import keras

from .contrastive import ContrastiveModel


class SimCLR(ContrastiveModel):
    """SimCLR model for self-supervised learning"""

    def __init__(
        self,
        encoder: keras.Model,
        projector: keras.Model,
        contrastive_augmenter: Callable[[keras.KerasTensor], keras.KerasTensor] | None = None,
        classification_augmenter: Callable[[keras.KerasTensor], keras.KerasTensor] | None = None,
        linear_probe: keras.Model | None = None,
        temperature: float = 0.1,
    ):
        super().__init__(
            encoder=encoder,
            projector=projector,
            contrastive_augmenter=contrastive_augmenter,
            classification_augmenter=classification_augmenter,
            linear_probe=linear_probe,
        )
        self.temperature = temperature

    def contrastive_loss(self, projections_1, projections_2):
        """Contrastive loss function for SimCLR"""
        # InfoNCE loss (information noise-contrastive estimation)
        # NT-Xent loss (normalized temperature-scaled cross entropy)

        # cosine similarity: the dot product of the l2-normalized feature vectors
        projections_1 = keras.ops.normalize(projections_1, axis=1)
        projections_2 = keras.ops.normalize(projections_2, axis=1)
        similarities = keras.ops.matmul(projections_1, keras.ops.transpose(projections_2)) / self.temperature

        # the temperature-scaled similarities are used as logits for cross-entropy
        batch_size = keras.ops.shape(projections_1)[0]
        contrastive_labels = keras.ops.arange(batch_size)
        loss1 = keras.losses.sparse_categorical_crossentropy(contrastive_labels, similarities, from_logits=True)
        loss2 = keras.losses.sparse_categorical_crossentropy(
            contrastive_labels, keras.ops.transpose(similarities), from_logits=True
        )
        return (loss1 + loss2) / 2
