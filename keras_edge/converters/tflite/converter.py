import io
import tempfile
from enum import StrEnum

import keras
import numpy as np
import numpy.typing as npt
import pandas as pd

import tensorflow as tf

from ..cpp import xxd_c_dump

class QuantizationType(StrEnum):
    FP32 = "FP32"
    FP16 = "FP16"
    INT8 = "INT8"
    INT16X8 = "INT16X8"

class TfLiteKerasConverter:

    def __init__(self,
        model: keras.Model,
    ):
        """TFLite Keras model converter that handles conversion, evaluation, and prediction.

        Args:
            model (keras.Model): Keras model
        """
        self.model = model
        self.representative_dataset = None
        self._converter: tf.lite.TFLiteConverter|None = None
        self._tflite_content: str|None = None
        self.tf_model_path = tempfile.TemporaryDirectory()

    def convert(
        self,
        test_x: npt.NDArray | None = None,
        quantization: QuantizationType = QuantizationType.FP32,
        io_type: str|None = None,
        use_concrete: bool = False,
        strict: bool = True
    ) -> str:
        """Convert TF model into TFLite model content

        Args:
            test_x (npt.NDArray | None, optional): Test dataset. Defaults to None.
            quantization (QuantizationType, optional): Quantization type. Defaults to QuantizationType.FP32.
            io_type (str | None, optional): Input/Output type. Defaults to None.
            use_concrete (bool, optional): Use concrete function. Defaults to False.
            strict (bool, optional): Strict mode. Defaults to True.

        Returns:
            str: TFLite content
        """
        quantization = QuantizationType(quantization)

        feat_shape = self.model.input_shape[1:]
        input_shape = (1,) + feat_shape # Add 1 for batch dimension
        input_spec = tf.TensorSpec(shape=input_shape, dtype=self.model.input_dtype)

        # Following is a workaround for bug (https://github.com/tensorflow/tflite-micro/issues/2319)
        # Default TFLiteConverter generates equivalent graph w/ SpaceToBatchND operations but losses dilation_rate factor.
        if use_concrete:
            model_func = tf.function(func=self.model)
            model_cf = model_func.get_concrete_function(input_spec)
            converter = tf.lite.TFLiteConverter.from_concrete_functions([model_cf])
        else:
            self.model.export(self.tf_model_path.name, format="tf_saved_model")
            converter = tf.lite.TFLiteConverter.from_saved_model(self.tf_model_path.name)
            # Following is broken...hence that workaround
            # converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
        # END IF

        if test_x is None:
            test_x = np.random.rand(1000, *feat_shape)

        def rep_dataset():
            """Helper function to generate representative dataset"""
            for i in range(test_x.shape[0]):
                yield [test_x[i : i + 1]]

        self.representative_dataset = rep_dataset

        match quantization:
            # float32 weights, bias, activation
            case QuantizationType.FP32:
                pass
            # float16 weights, bias, activation
            case QuantizationType.FP16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
            # int8 weights, bias, activation
            case QuantizationType.INT8:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                io_dtype = tf.dtypes.as_dtype(io_type) if io_type else tf.int8
                converter.inference_input_type = io_dtype
                converter.inference_output_type = io_dtype
                converter.representative_dataset = self.representative_dataset
            # int8 weights, int64 bias, int16 activation
            case QuantizationType.INT16X8:
                converter.representative_dataset = self.representative_dataset
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
        # END MATCH

        # For fallback append tf.lite.OpsSet.TFLITE_BUILTINS for INT8 and INT16X8
        if not strict and quantization in [QuantizationType.INT8, QuantizationType.INT16X8]:
            converter.target_spec.supported_ops.append(tf.lite.OpsSet.TFLITE_BUILTINS)

        # Convert model
        self._converter = converter
        self._tflite_content = converter.convert()

        return self._tflite_content

    def debug_quantization(self) -> pd.DataFrame:
        """Debug quantized TFLite model content.
        """

        if self._converter is None:
            raise ValueError("No TFLite content to debug. Run convert() first.")

        if self.representative_dataset is None:
            raise ValueError("No representative dataset provided. Run convert() with test_x first.")

        # Debug model
        debugger = tf.lite.experimental.QuantizationDebugger(
            converter=self._converter,
            debug_dataset=self.representative_dataset
        )
        debugger.run()

        with io.StringIO() as f:
            debugger.layer_statistics_dump(f)
            f.seek(0)
            layer_stats = pd.read_csv(f)
        # END WITH

        # Add custom metrics
        layer_stats["range"] = 255.0 * layer_stats["scale"]
        layer_stats["rmse/scale"] = layer_stats.apply(
            lambda row: np.sqrt(row["mean_squared_error"]) / row["scale"], axis=1
        )
        return layer_stats

    def predict(
        self,
        x: npt.NDArray,
        input_name: str | None = None,
        output_name: str | None = None,
    ):
        # Prepare the test data
        inputs = x.copy()
        inputs = inputs.astype(np.float32)

        interpreter = tf.lite.Interpreter(model_content=self._tflite_content)
        interpreter.allocate_tensors()

        # No signature
        if len(interpreter.get_signature_list()) == 0:
            output_details = interpreter.get_output_details()[0]
            input_details = interpreter.get_input_details()[0]

            input_scale: list[float] = input_details["quantization_parameters"]["scales"]
            input_zero_point: list[int] = input_details["quantization_parameters"][
                "zero_points"
            ]
            output_scale: list[float] = output_details["quantization_parameters"]["scales"]
            output_zero_point: list[int] = output_details["quantization_parameters"][
                "zero_points"
            ]

            inputs = inputs.reshape([-1] + input_details["shape_signature"].tolist())
            if len(input_scale) and len(input_zero_point):
                inputs = inputs / input_scale[0] + input_zero_point[0]
                inputs = inputs.astype(input_details["dtype"])

            outputs = []
            for sample in inputs:
                interpreter.set_tensor(input_details['index'], sample)
                interpreter.invoke()
                y = interpreter.get_tensor(output_details['index'])
                outputs.append(y)
            outputs = np.concatenate(outputs, axis=0)

            if len(output_scale) and len(output_zero_point):
                outputs = outputs.astype(np.float32)
                outputs = (outputs - output_zero_point[0]) * output_scale[0]

            return outputs

        # WITH Signature
        model_sig = interpreter.get_signature_runner()
        inputs_details = model_sig.get_input_details()
        outputs_details = model_sig.get_output_details()
        if input_name is None:
            input_name = list(inputs_details.keys())[0]
        if output_name is None:
            output_name = list(outputs_details.keys())[0]
        input_details = inputs_details[input_name]
        output_details = outputs_details[output_name]
        input_scale: list[float] = input_details["quantization_parameters"]["scales"]
        input_zero_point: list[int] = input_details["quantization_parameters"][
            "zero_points"
        ]
        output_scale: list[float] = output_details["quantization_parameters"]["scales"]
        output_zero_point: list[int] = output_details["quantization_parameters"][
            "zero_points"
        ]

        inputs = inputs.reshape([-1] + input_details["shape_signature"].tolist()[1:])
        if len(input_scale) and len(input_zero_point):
            inputs = inputs / input_scale[0] + input_zero_point[0]
            inputs = inputs.astype(input_details["dtype"])

        outputs = np.array(
            [
                model_sig(**{input_name: inputs[i : i + 1]})[output_name][0]
                for i in range(inputs.shape[0])
            ],
            dtype=output_details["dtype"],
        )

        if len(output_scale) and len(output_zero_point):
            outputs = outputs.astype(np.float32)
            outputs = (outputs - output_zero_point[0]) * output_scale[0]

        return outputs

    def evaluate(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        input_name: str | None = None,
        output_name: str | None = None,
    ) -> npt.NDArray:
        """Evaluate TFLite model

        Args:
            x (npt.NDArray): Input samples
            y (npt.NDArray): Input labels
            input_name (str | None, optional): Input layer name. Defaults to None.
            output_name (str | None, optional): Output layer name. Defaults to None.

        Returns:
            npt.NDArray: Loss values
        """
        y_pred = self.predict(
            x=x,
            input_name=input_name,
            output_name=output_name,
        )
        loss_function = keras.losses.get(self.model.loss)
        loss = loss_function(y, y_pred).numpy()
        return loss

    def export(self, tflite_path: str):
        """Export TFLite model content to file

        Args:
            tflite_path (str): TFLite file path
        """
        if self._tflite_content is None:
            raise ValueError("No TFLite content to export. Run convert() first.")

        with open(tflite_path, "wb") as f:
            f.write(self._tflite_content)

    def export_header(self, header_path: str, name: str = "model"):
        """Export TFLite model as C header file.

        Args:
            header_path (str): Header file path
            name (str, optional): Variable name. Defaults to "model".
        """
        with tempfile.NamedTemporaryFile() as f:
            self.export(f.name)
            xxd_c_dump(
                src_path=f.name,
                dst_path=header_path,
                var_name=name,
                chunk_len=20,
                is_header=True
            )
        # END WITH

    def cleanup(self):
        """Cleanup temporary files"""
        self.tf_model_path.cleanup()

class TfLiteKerasInterpreter:

    def __init__(
        self,
        model_content: str,
        input_name: str | None = None,
        output_name: str | None = None,
        signature_key: str | None = None,
    ):
        """TFLite model interpreter that takes care of I/O conversion and prediction.

        Args:
            model_content (str): TFLite model content
            input_name (str | None, optional): Input layer name. Defaults to None.
            output_name (str | None, optional): Output layer name. Defaults to None.
            signature_key (str | None, optional): Signature key. Defaults to None
        """
        self.model_content = model_content
        self.interpreter = tf.lite.Interpreter(model_content=model_content)
        self.interpreter.allocate_tensors()

        self.signature_key = signature_key
        self._has_signature = False

        self._input_name = input_name
        self._input_shape = None
        self._input_scale = None
        self._input_zero_point = None
        self._input_dtype = "float32"

        self._output_name = output_name
        self._output_scale = None
        self._output_zero_point = None
        self._output_dtype = "float32"

    def compile(self):
        """Compile model and extract input/output details."""

        # Some models may lose signature after converting to tflite due to TF issues.
        # Most prevalent for models lowered to concrete functions.
        self._has_signature = len(self.interpreter.get_signature_list()) > 0

        if not self._has_signature:
            input_details = self.interpreter.get_input_details()[0]
            output_details = self.interpreter.get_output_details()[0]
            self._input_shape = input_details["shape_signature"].tolist()
            self._input_name = input_details['index']
            self._output_name = output_details['index']

        else:
            model_sig = self.interpreter.get_signature_runner(self.signature_key)
            inputs_details = model_sig.get_input_details()
            outputs_details = model_sig.get_output_details()
            if self._input_name is None:
                self._input_name = list(inputs_details.keys())[0]
            if self._output_name is None:
                self._output_name = list(outputs_details.keys())[0]
            input_details = inputs_details[self._input_name]
            output_details = outputs_details[self._output_name]
            self._input_shape = input_details["shape_signature"].tolist()[1:]
        # END IF

        input_scale: list[float] = input_details["quantization_parameters"]["scales"]
        input_zero_point: list[int] = input_details["quantization_parameters"]["zero_points"]
        output_scale: list[float] = output_details["quantization_parameters"]["scales"]
        output_zero_point: list[int] = output_details["quantization_parameters"]["zero_points"]

        self._input_dtype = input_details["dtype"]
        if len(input_scale) and len(input_zero_point):
            self._input_scale = input_scale[0]
            self._input_zero_point = input_zero_point[0]
        # END IF

        self._output_dtype = output_details["dtype"]
        if len(output_scale) and len(output_zero_point):
            self._output_scale = output_scale[0]
            self._output_zero_point = output_zero_point[0]
        # END IF

    def convert_input(
        self,
        x: npt.NDArray
    ) -> npt.NDArray:
        """Convert input data based on quantization.

        NOTE: predict() will call this method internally.

        Args:
            x (npt.NDArray): Input samples

        Returns:
            npt.NDArray: Prepared input samples
        """
        inputs = x.copy()
        inputs = inputs.reshape([-1] + self._input_shape)
        if self._input_scale and self._input_zero_point:
            inputs = inputs / self._input_scale + self._input_zero_point
        inputs = inputs.astype(self._input_dtype)
        return inputs

    def convert_output(
        self,
        outputs: npt.NDArray
    ) -> npt.NDArray:
        """Convert output data based on quantization.

        NOTE: predict() will call this method internally.

        Args:
            outputs (npt.NDArray): Output samples

        Returns:
            npt.NDArray: Prepared output samples
        """
        outputs = outputs.astype(self._output_dtype)
        if self._output_scale and self._output_zero_point:
            outputs = (outputs - self._output_zero_point) * self._output_scale
        return outputs

    def predict(
        self,
        x: npt.NDArray,
    ) -> npt.NDArray:
        """Predict using TFLite model

        Args:
            x (npt.NDArray): Input samples

        Returns:
            npt.NDArray: Predicted values
        """
        inputs = self.convert_input(x)

        if not self._has_signature:
            outputs = []
            for sample in inputs:
                self.interpreter.set_tensor(self._input_name, sample)
                self.interpreter.invoke()
                y = self.interpreter.get_tensor(self._output_name)
                outputs.append(y)
            outputs = np.concatenate(outputs, axis=0)
        else:
            model_sig = self.interpreter.get_signature_runner(self.signature_key)
            outputs = np.array([
                model_sig(**{self._input_name: inputs[i : i + 1]})[self._output_name][0]
                for i in range(inputs.shape[0])
            ], dtype=self._output_dtype)
        # END IF

        outputs = self.convert_output(outputs)

        return outputs
