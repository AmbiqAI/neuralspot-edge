# Home

<p align="center">
  <a href="https://github.com/AmbiqAI/neuralspot-edge"><img src="./assets/nse-banner.png" alt="NSE"></a>
</p>

---

**Documentation**: <a href="https://ambiqai.github.io/neuralspot-edge" target="_blank">https://ambiqai.github.io/neuralspot-edge</a>

**Source Code**: <a href="https://github.com/AmbiqAI/neuralspot-edge" target="_blank">https://github.com/AmbiqAI/neuralspot-edge</a>

---


<img style="float: left; margin: 0px 15px 15px 0px;" src="./assets/nse-logo-128.png" width="96px" />

neuralSPOT Edge (**NSE**) is [Keras 3](https://keras.io) add-on focused on training and deploying models on resource-constrained, edge devices. NSE relies heavily on [Keras 3](https://keras.io) leveraging it's multi-backend support and customizable architecture. This package provides a variety of additional models, layers, optimizers, quantizers, and other components to help users train and deploy models for edge devices.

<br style="clear:both" />

<div class="grid cards" markdown>

- :material-rocket-launch: [Getting Started](usage/index.md)
- :material-api: [API Documentation](api/neuralspot_edge)
- :simple-docsdotrs: [Usage Examples](examples/index.md)
- :simple-jupyter: [Explore Guides](guides/index.md)

</div>

## Main Features

* [**Callbacks**](api/neuralspot_edge/callbacks): Training callbacks
* [**Converters**](api/neuralspot_edge/converters): Converters for exporting models
* [**Interpreters**](api/neuralspot_edge/interpreters): Inference engine interpreters (e.g. TFLite)
* [**Layers**](api/neuralspot_edge/layers): Custom layers including `tf.data.Dataset` preprocessing layers
* [**Losses**](api/neuralspot_edge/losses): Additional losses such as SimCLRLoss
* [**Metrics**](api/neuralspot_edge/metrics): Custom metrics such as SNR
* [**Models**](api/neuralspot_edge/models): Highly parameterized 1D/2D model architectures
* [**Optimizers**](api/neuralspot_edge/optimizers): Additional optimizers
* [**Plotting**](api/neuralspot_edge/plotting): Plotting routines
* [**Quantizers**](api/neuralspot_edge/quantizers): Quantization techniques
* [**Trainers**](api/neuralspot_edge/trainers): Custom trainers such as SSL contrastive learning
* [**Utils**](api/neuralspot_edge/utils): Utility functions

## Problems **NSE** looks to solve

### Compatability issues between frameworks and inference engines

- [x] By leveraging Keras 3, entire workflows can be run using a variety of backends using a consistent front-end API. This allows selecting a backend that plays nicely with a specific inference engine without rewriting the entire model.

### SOTA models dont scale down well and come in limited configurations

- [x] By providing highly parameterized model architectures based on SOTA models, users can easily scale down models to fit their needs.

### Limited 1D time-series models

- [x] Most included models in NSE provide both 1D and 2D versions. The package also contains time-series specific models.

### Limited support for quantization, pruning, and other model optimization techniques

- [x] NSE provides a variety of quantization and pruning techniques to optimize models for edge deployment.
