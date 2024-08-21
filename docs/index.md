# Home

neuralSPOT Edge (**NSE**) is [Keras 3](https://keras.io) add-on focused on training and deploying models on resource-constrained, edge devices. It relies heavily on [Keras 3](https://keras.io) leveraging it's multi-backend support and customizable architecture. NSEdge provides a variety of models, layers, optimizers, quantizers, and other components to help users train and deploy models on edge devices.

<div class="grid cards" markdown>

- :material-rocket-launch: [Getting Started](usage/index.md)
- :material-api: [API Documentation](api/index.md)
- :simple-docsdotrs: [Usage Examples](examples/index.md)
- :simple-jupyter: [Explore Guides](guides/index.md)

</div>

## Main Features

* [**Models**](api/models/index.md): Highly parameterized 1D/2D model architectures
* [**Layers**](api/layers/index.md): Custom layers including `tf.data.Dataset` preprocessing layers
* [**Trainers**](api/trainers/index.md): Custom trainers such as SSL contrastive learning
* [**Optimizers**](api/optimizers/index.md): Additional optimizers
* [**Quantizers**](api/quantizers/index.md): Quantization techniques
* [**Metrics**](api/metrics/index.md): Custom metrics such as SNR
* [**Losses**](api/losses/index.md): Additional losses such as SimCLRLoss
* [**Converters**](api/converters/index.md): Converters for exporting models
* [**Interpreters**](api/interpreters/index.md): Inference engine interpreters (e.g. TFLite)
* [**Callbacks**](api/callbacks/index.md): Training callbacks


## Problems this add-on looks to solve


### P1. Compatability issues between frameworks and inference engines

> S1. By leveraging Keras 3, entire workflows can be run using a vareity of backends that play nicer with certain inference engines. Since Keras provides intermedite representations, it is easier to convert between formats.

### P2. SOTA models dont scale down well and come in limited configurations

> S2. By providing highly parameterized model architectures based on SOTA models, users can easily scale down models to fit their needs.

### P3. Limited time-series 1D models

> S3. Most included models provide both 1D and 2D versions. NSE also containes time-series specific models.

### P4. Limited support for quantization, pruning, and other model optimization techniques

> S4. NSE provides a variety of quantization and pruning techniques to optimize models for edge deployment.
