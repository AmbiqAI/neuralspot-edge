# Interpreters

Interpreters are classes that are used to interpret pre-trained model files such as TensorFlow and TensorFlow Lit. These classes are used to load the model file and provide a consistent interface for making predictions. Often a `nse.converter` is used to convert the model and `nse.interpreter` is used verify the model results.

## Available Interpreters

* [TfLiteKerasInterpreter](./tflite.md#nseinterpreterstflitetflitekerasinterpreter)