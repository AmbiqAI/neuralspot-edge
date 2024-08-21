# Getting Started

## Install neuralspot-edge

You can install __neuralspot-edge__ from PyPI via:

```bash
pip install neuralspot-edge
```

Alternatively, you can install using Poetry via:

```bash
poetry add neuralspot-edge
```

!!! note
    __neuralspot-edge__ relies heavily on [Keras 3](https://keras.io/). Since Keras supports multiple backend frameworks, you'll need to install one of the following. By default, we assume __TensorFlow__ as a number of the data pipelines are built using TensorFlow `tf.data.Dataset`.

    * [Installing JAX](https://jax.readthedocs.io/en/latest/installation.html)
    * [Installing TensorFlow](https://www.tensorflow.org/install)
    * [Installing PyTorch](https://pytorch.org/get-started/locally/)

## Requirements

* [Python ^3.11+](https://www.python.org)

Check the project's [pyproject.toml](https://github.com/AmbiqAI/neuralspot-edge/blob/main/pyproject.toml) file for a list of up-to-date Python dependencies. Note that the installation methods above will install all required dependencies.
