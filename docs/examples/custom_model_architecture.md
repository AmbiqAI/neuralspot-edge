# Making Custom Model Architecture

## Introduction

This guide will show you how to create a custom model architecture in a similar fashion to the models provided in the `nse.models` module. This guide will cover the following topics:

## Setup

```py linenums="1"

import keras
import neuralspot_edge as nse
from pydantic import BaseModel, Field
```

## Define Model Parameters

The first step is to define the parameters for the model. The preferred way to do this is to use Pydantic models or dataclasses. Pydantic models allow you to define the structure of the parameters and provide validation. Below is an example of how to define the parameters for a custom model. This model is essentially a series of fully connected layers with a specified number of units and activation function for each layer.


```py linenums="1"

class CustomLayerParams(BaseModel):
    """Custom layer parameters

    Attributes:
        units (int): Number of units in the layer
        activation (str): Activation function for the layer
    """
    units: int = Field(..., description="Number of units in the layer")
    activation: str = Field("relu", description="Activation function for the layer")

class CustomModelParams(BaseModel):
    """Custom model parameters

    Attributes:
        layers (list[CustomLayerParams]): List of layer parameters
    """
    layers: list[CustomLayerParams] = Field(..., description="List of layer parameters")

```

## Create Functional Layer

The next step is to create a functional layer representation. In **NSE**, we prefer using functional layers/models that given a set of parameters returns a closure that can be used to create the layer/model. Below is an example of how to create a functional layer for the custom model. This function takes an input tensor, the parameters for the model, and the number of classes (if applicable) and returns the output tensor.

We dont directly return a `keras.Model` as this allows combining a variety of topologies if desired.

```py linenums="1"

    def custom_model_layer(
        x: keras.KerasTensor,
        params: CustomLayerParams,
        num_classes: int | None = None,
    ) -> keras.KerasTensor:

        y = x
        # Create fully connected network from params
        for layer in params.layers:
            y = keras.layers.Dense(layer.units, activation=layer.activation)(y)

        if num_classes:
            y = keras.layers.Dense(num_classes, activation="softmax")(y)

        return y
```


## Create Model Generator Class

Lastly, we create a class that generates the model from the parameters. This class will have a static method that takes the input tensor, the parameters for the model, and the number of classes (if applicable) and returns the model. This class can be used to generate the model from the parameters.

It's also possible to subclass `keras.Model` and override the `__init__` method to take the parameters and create the model. This is useful if you want to have a custom model class that can be used like a regular Keras model. In **NSE**, we try to reserve subclassing `keras.Model` for Trainers (e.g. `nse.trainers.SimCLRTrainer`) and use functional layers for models.

```py linenums="1"

class CustomModel:
    """Helper class to generate model from parameters"""

    @staticmethod
    def layer_from_params(inputs: keras.Input, params: CustomModelParams | dict, num_classes: int | None = None):
        """Create functional layer from parameters"""
        if isinstance(params, dict):
            params = CustomModelParams(**params)
        return custom_model_layer(x=inputs, params=params, num_classes=num_classes)

    @staticmethod
    def model_from_params(inputs: keras.Input, params: CustomModelParams | dict, num_classes: int | None = None):
        """Create model from parameters"""
        outputs = CustomModel.layer_from_params(inputs=inputs, params=params, num_classes=num_classes)
        return keras.Model(inputs=inputs, outputs=outputs)

```

## Usage

Now that we have defined the model parameters, the functional layer, and the model generator class, we can use them to create a custom model. Below is an example of how to create a custom model using the defined parameters.

```py linenums="1"

# Define input tensor
inputs = keras.Input(shape=(32,))

# Define model parameters
params = CustomModelParams(
    layers=[
        CustomLayerParams(units=64, activation="relu"),
        CustomLayerParams(units=32, activation="relu"),
    ]
)

# Create model
model = CustomModel.model_from_params(inputs=inputs, params=params, num_classes=10)

# Print model summary
model.summary()

x = keras.random.normal((1, 32), dtype="float32")
y = model(x)

```
