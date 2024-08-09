from typing import Literal
import keras
from keras.src.metrics import metrics_utils


def compute_metrics(metrics: list[keras.Metric], y_true, y_pred) -> dict[str, float]:
    """Compute set of metrics for y_true and y_pred.

    Args:
        metrics (list[keras.Metric]): List of metrics
        y_true: True y labels
        y_pred: Predicted y labels

    Returns:
        dict: Dictionary of metric names and values
    """
    results = {}
    for metric in metrics:
        metric.reset_state()
        metric.update_state(y_true, y_pred)
        results[metric.name] = metric.result().numpy()
    return results


def confusion_matrix(
    labels,
    predictions,
    num_classes,
    weights=None,
    dtype="int32",
    normalize: Literal["true", "pred", "all"] | None = None,
) -> keras.KerasTensor:
    """Compute confusion matrix using keras w/ addition to normalize.

    Args:
        labels: Ground truth values.
        predictions: The predicted values.
        num_classes: The number of classes.
        weights: Sample weights.
        dtype: Data type.
        normalize: Normalization mode.

    Returns:
        keras.KerasTensor: Confusion matrix
    """
    cm = metrics_utils.confusion_matrix(
        labels,
        predictions,
        num_classes,
        weights=weights,
        dtype=dtype,
    )

    if normalize == "true":
        cm = keras.ops.divide(cm, keras.ops.sum(cm, axis=1, keepdims=True))
    elif normalize == "pred":
        cm = keras.ops.divide(cm, keras.ops.sum(cm, axis=0, keepdims=True))
    elif normalize == "all":
        cm = keras.ops.divide(cm, keras.ops.sum(cm))
    cm = keras.ops.nan_to_num(cm, posinf=0, neginf=0)
    return cm
