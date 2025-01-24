import flatbuffers
import numpy as np
from tensorflow.lite.python import schema_py_generated as tflite

def convert_float32_buffer_to_float16(data: bytes) -> bytes:
    """Convert raw float32 bytes to float16 bytes using NumPy."""
    if not data:
        return data
    float32_array = np.frombuffer(data, dtype=np.float32)
    float16_array = float32_array.astype(np.float16)
    return float16_array.tobytes()

def read_options_from_operator(op, typed_class):
    """Helper to initialize `typed_class` (e.g. Conv2DOptions) from op.BuiltinOptions()."""
    opts = typed_class()
    if op.BuiltinOptions() is None:
        return None
    opts.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
    return opts

def maybe_fix_quantized_bias_type(builder_func_add, old_bias_type):
    """
    If old_bias_type is FLOAT32, return FLOAT16, else return the original.
    We call builder_func_add(builder, type) to set it.
    """
    if old_bias_type == tflite.TensorType.FLOAT32:
        return tflite.TensorType.FLOAT16
    return old_bias_type

def copy_builtin_options(builder, op):
    """
    Reads the builtin_options from `op` and copies all fields.
    Also sets `quantized_bias_type` from FLOAT32 to FLOAT16 if present.
    """
    builtin_type = op.BuiltinOptionsType()
    if builtin_type == tflite.BuiltinOptions.NONE:
        return 0

    # ------------------------------------------------------------------------
    # Conv2DOptions
    if builtin_type == tflite.BuiltinOptions.Conv2DOptions:
        old_opts = read_options_from_operator(op, tflite.Conv2DOptions)
        if not old_opts:
            return 0
        tflite.Conv2DOptionsStart(builder)
        tflite.Conv2DOptionsAddPadding(builder, old_opts.Padding())
        tflite.Conv2DOptionsAddStrideW(builder, old_opts.StrideW())
        tflite.Conv2DOptionsAddStrideH(builder, old_opts.StrideH())
        tflite.Conv2DOptionsAddFusedActivationFunction(builder, old_opts.FusedActivationFunction())
        tflite.Conv2DOptionsAddDilationWFactor(builder, old_opts.DilationWFactor())
        tflite.Conv2DOptionsAddDilationHFactor(builder, old_opts.DilationHFactor())
        # If the schema has quantized_bias_type, we switch it to FLOAT16 if originally FLOAT32
        if hasattr(old_opts, 'QuantizedBiasType'):
            new_bias_type = maybe_fix_quantized_bias_type(tflite.Conv2DOptionsAddQuantizedBiasType,
                                                          old_opts.QuantizedBiasType())
            tflite.Conv2DOptionsAddQuantizedBiasType(builder, new_bias_type)
        return tflite.Conv2DOptionsEnd(builder)

    # DepthwiseConv2DOptions
    elif builtin_type == tflite.BuiltinOptions.DepthwiseConv2DOptions:
        old_opts = read_options_from_operator(op, tflite.DepthwiseConv2DOptions)
        if not old_opts:
            return 0
        tflite.DepthwiseConv2DOptionsStart(builder)
        tflite.DepthwiseConv2DOptionsAddPadding(builder, old_opts.Padding())
        tflite.DepthwiseConv2DOptionsAddStrideW(builder, old_opts.StrideW())
        tflite.DepthwiseConv2DOptionsAddStrideH(builder, old_opts.StrideH())
        tflite.DepthwiseConv2DOptionsAddDepthMultiplier(builder, old_opts.DepthMultiplier())
        tflite.DepthwiseConv2DOptionsAddFusedActivationFunction(builder, old_opts.FusedActivationFunction())
        tflite.DepthwiseConv2DOptionsAddDilationWFactor(builder, old_opts.DilationWFactor())
        tflite.DepthwiseConv2DOptionsAddDilationHFactor(builder, old_opts.DilationHFactor())
        # Fix quantized_bias_type if the field exists
        if hasattr(old_opts, 'QuantizedBiasType'):
            new_bias_type = maybe_fix_quantized_bias_type(tflite.DepthwiseConv2DOptionsAddQuantizedBiasType,
                                                          old_opts.QuantizedBiasType())
            tflite.DepthwiseConv2DOptionsAddQuantizedBiasType(builder, new_bias_type)
        return tflite.DepthwiseConv2DOptionsEnd(builder)

    # Pool2DOptions
    elif builtin_type == tflite.BuiltinOptions.Pool2DOptions:
        old_opts = read_options_from_operator(op, tflite.Pool2DOptions)
        if not old_opts:
            return 0
        tflite.Pool2DOptionsStart(builder)
        tflite.Pool2DOptionsAddPadding(builder, old_opts.Padding())
        tflite.Pool2DOptionsAddStrideW(builder, old_opts.StrideW())
        tflite.Pool2DOptionsAddStrideH(builder, old_opts.StrideH())
        tflite.Pool2DOptionsAddFilterWidth(builder, old_opts.FilterWidth())
        tflite.Pool2DOptionsAddFilterHeight(builder, old_opts.FilterHeight())
        tflite.Pool2DOptionsAddFusedActivationFunction(builder, old_opts.FusedActivationFunction())
        return tflite.Pool2DOptionsEnd(builder)

    # FullyConnectedOptions
    elif builtin_type == tflite.BuiltinOptions.FullyConnectedOptions:
        old_opts = read_options_from_operator(op, tflite.FullyConnectedOptions)
        if not old_opts:
            return 0
        tflite.FullyConnectedOptionsStart(builder)
        tflite.FullyConnectedOptionsAddFusedActivationFunction(builder, old_opts.FusedActivationFunction())
        tflite.FullyConnectedOptionsAddWeightsFormat(builder, old_opts.WeightsFormat())
        tflite.FullyConnectedOptionsAddKeepNumDims(builder, old_opts.KeepNumDims())
        # If the schema version includes quantized_bias_type
        if hasattr(old_opts, 'QuantizedBiasType'):
            new_bias_type = maybe_fix_quantized_bias_type(tflite.FullyConnectedOptionsAddQuantizedBiasType,
                                                          old_opts.QuantizedBiasType())
            tflite.FullyConnectedOptionsAddQuantizedBiasType(builder, new_bias_type)
        return tflite.FullyConnectedOptionsEnd(builder)

    # AddOptions
    elif builtin_type == tflite.BuiltinOptions.AddOptions:
        old_opts = read_options_from_operator(op, tflite.AddOptions)
        if not old_opts:
            return 0
        tflite.AddOptionsStart(builder)
        tflite.AddOptionsAddFusedActivationFunction(builder, old_opts.FusedActivationFunction())
        return tflite.AddOptionsEnd(builder)

    # MulOptions
    elif builtin_type == tflite.BuiltinOptions.MulOptions:
        old_opts = read_options_from_operator(op, tflite.MulOptions)
        if not old_opts:
            return 0
        tflite.MulOptionsStart(builder)
        tflite.MulOptionsAddFusedActivationFunction(builder, old_opts.FusedActivationFunction())
        return tflite.MulOptionsEnd(builder)

    # SubOptions
    elif builtin_type == tflite.BuiltinOptions.SubOptions:
        old_opts = read_options_from_operator(op, tflite.SubOptions)
        if not old_opts:
            return 0
        tflite.SubOptionsStart(builder)
        tflite.SubOptionsAddFusedActivationFunction(builder, old_opts.FusedActivationFunction())
        return tflite.SubOptionsEnd(builder)

    # SoftmaxOptions
    elif builtin_type == tflite.BuiltinOptions.SoftmaxOptions:
        old_opts = read_options_from_operator(op, tflite.SoftmaxOptions)
        if not old_opts:
            return 0
        tflite.SoftmaxOptionsStart(builder)
        tflite.SoftmaxOptionsAddBeta(builder, old_opts.Beta())
        return tflite.SoftmaxOptionsEnd(builder)

    # ReshapeOptions
    elif builtin_type == tflite.BuiltinOptions.ReshapeOptions:
        old_opts = read_options_from_operator(op, tflite.ReshapeOptions)
        if not old_opts:
            return 0
        shape_len = old_opts.NewShapeLength()
        new_shape_offset = 0
        if shape_len > 0:
            shape_list = [old_opts.NewShape(i) for i in range(shape_len)]
            tflite.ReshapeOptionsStartNewShapeVector(builder, shape_len)
            for dim in reversed(shape_list):
                builder.PrependInt32(dim)
            new_shape_offset = builder.EndVector(shape_len)

        tflite.ReshapeOptionsStart(builder)
        if new_shape_offset:
            tflite.ReshapeOptionsAddNewShape(builder, new_shape_offset)
        return tflite.ReshapeOptionsEnd(builder)

    # ResizeBilinearOptions
    elif builtin_type == tflite.BuiltinOptions.ResizeBilinearOptions:
        old_opts = read_options_from_operator(op, tflite.ResizeBilinearOptions)
        if not old_opts:
            return 0
        tflite.ResizeBilinearOptionsStart(builder)
        tflite.ResizeBilinearOptionsAddNewHeight(builder, old_opts.NewHeight())
        tflite.ResizeBilinearOptionsAddNewWidth(builder, old_opts.NewWidth())
        tflite.ResizeBilinearOptionsAddAlignCorners(builder, old_opts.AlignCorners())
        tflite.ResizeBilinearOptionsAddHalfPixelCenters(builder, old_opts.HalfPixelCenters())
        return tflite.ResizeBilinearOptionsEnd(builder)

    # ResizeNearestNeighborOptions
    elif builtin_type == tflite.BuiltinOptions.ResizeNearestNeighborOptions:
        old_opts = read_options_from_operator(op, tflite.ResizeNearestNeighborOptions)
        if not old_opts:
            return 0
        tflite.ResizeNearestNeighborOptionsStart(builder)
        tflite.ResizeNearestNeighborOptionsAddNewHeight(builder, old_opts.NewHeight())
        tflite.ResizeNearestNeighborOptionsAddNewWidth(builder, old_opts.NewWidth())
        tflite.ResizeNearestNeighborOptionsAddAlignCorners(builder, old_opts.AlignCorners())
        tflite.ResizeNearestNeighborOptionsAddHalfPixelCenters(builder, old_opts.HalfPixelCenters())
        return tflite.ResizeNearestNeighborOptionsEnd(builder)

    # ------------------------------------------------------------------------
    # Add more ops if your model uses them (e.g. BatchedMatMul, etc.).
    # ------------------------------------------------------------------------

    # If we didn't match anything, just return 0
    return 0

def convert_flatbuffer_float32_to_float16(old_data):
    """
    Reads `input_tflite`, converts float32 tensors -> float16,
    updates quantized_bias_type from FLOAT32->FLOAT16,
    and writes the modified model to `output_tflite`.
    """


    # Parse model
    old_model = tflite.Model.GetRootAsModel(old_data, 0)

    # Prepare builder
    builder = flatbuffers.Builder(initialSize=len(old_data) + 1024)

    # ----- A) Copy Buffers (with optional float32->float16 conversion) -----
    new_buffer_data_list = []
    for i in range(old_model.BuffersLength()):
        old_buf = old_model.Buffers(i)
        if old_buf.DataLength() > 0:
            new_buffer_data_list.append(old_buf.DataAsNumpy().tobytes())
        else:
            new_buffer_data_list.append(b"")

    # Identify which buffers are float32
    float32_buffer_indices = set()
    for sg_i in range(old_model.SubgraphsLength()):
        sg = old_model.Subgraphs(sg_i)
        for t_i in range(sg.TensorsLength()):
            tensor = sg.Tensors(t_i)
            if tensor.Type() == tflite.TensorType.FLOAT32:
                float32_buffer_indices.add(tensor.Buffer())

    # Convert buffers
    for idx in float32_buffer_indices:
        old_bytes = new_buffer_data_list[idx]
        if old_bytes:
            new_buffer_data_list[idx] = convert_float32_buffer_to_float16(old_bytes)

    # ----- B) Rebuild model in new FlatBuffer -----

    # (1) OperatorCodes
    op_code_offsets = []
    for i in range(old_model.OperatorCodesLength()):
        op_code = old_model.OperatorCodes(i)
        tflite.OperatorCodeStart(builder)
        tflite.OperatorCodeAddBuiltinCode(builder, op_code.BuiltinCode())
        tflite.OperatorCodeAddVersion(builder, op_code.Version())
        if op_code.CustomCode():
            custom_code_str = op_code.CustomCode().decode('utf-8')
            cc_offset = builder.CreateString(custom_code_str)
            tflite.OperatorCodeAddCustomCode(builder, cc_offset)
        op_code_offsets.append(tflite.OperatorCodeEnd(builder))

    # (2) Buffers
    buffer_offsets = []
    for data_bytes in new_buffer_data_list:
        if data_bytes:
            data_offset = builder.CreateByteVector(data_bytes)
        else:
            data_offset = 0
        tflite.BufferStart(builder)
        tflite.BufferAddData(builder, data_offset)
        buffer_offsets.append(tflite.BufferEnd(builder))

    # (3) Subgraphs
    subgraph_offsets = []
    for sg_i in range(old_model.SubgraphsLength()):
        sg = old_model.Subgraphs(sg_i)

        # Tensors
        tensor_offsets = []
        for t_i in range(sg.TensorsLength()):
            tensor = sg.Tensors(t_i)
            name_offset = 0
            if tensor.Name():
                name_offset = builder.CreateString(tensor.Name().decode("utf-8"))

            # Build shape vector
            shape_vec = [tensor.Shape(d_i) for d_i in range(tensor.ShapeLength())]
            tflite.TensorStartShapeVector(builder, len(shape_vec))
            for dim in reversed(shape_vec):
                builder.PrependInt32(dim)
            shape_offset = builder.EndVector(len(shape_vec))

            new_type = tensor.Type()
            # If it's FLOAT32 and in our set of float32 buffers, switch to FLOAT16
            if new_type == tflite.TensorType.FLOAT32 and tensor.Buffer() in float32_buffer_indices:
                new_type = tflite.TensorType.FLOAT16

            # Copy quantization
            quant_offset = 0
            if tensor.Quantization():
                old_quant = tensor.Quantization()
                scales = [old_quant.Scale(i) for i in range(old_quant.ScaleLength())]
                zero_points = [old_quant.ZeroPoint(i) for i in range(old_quant.ZeroPointLength())]

                scale_offset = 0
                if scales:
                    tflite.QuantizationParametersStartScaleVector(builder, len(scales))
                    for s in reversed(scales):
                        builder.PrependFloat32(s)
                    scale_offset = builder.EndVector(len(scales))

                zp_offset = 0
                if zero_points:
                    tflite.QuantizationParametersStartZeroPointVector(builder, len(zero_points))
                    for z in reversed(zero_points):
                        builder.PrependInt64(z)
                    zp_offset = builder.EndVector(len(zero_points))

                min_offset = 0
                max_offset = 0
                if old_quant.MinLength() > 0:
                    mins = [old_quant.Min(i) for i in range(old_quant.MinLength())]
                    tflite.QuantizationParametersStartMinVector(builder, len(mins))
                    for m in reversed(mins):
                        builder.PrependFloat32(m)
                    min_offset = builder.EndVector(len(mins))

                if old_quant.MaxLength() > 0:
                    maxs = [old_quant.Max(i) for i in range(old_quant.MaxLength())]
                    tflite.QuantizationParametersStartMaxVector(builder, len(maxs))
                    for mm in reversed(maxs):
                        builder.PrependFloat32(mm)
                    max_offset = builder.EndVector(len(maxs))

                tflite.QuantizationParametersStart(builder)
                if scale_offset:
                    tflite.QuantizationParametersAddScale(builder, scale_offset)
                if zp_offset:
                    tflite.QuantizationParametersAddZeroPoint(builder, zp_offset)
                if min_offset:
                    tflite.QuantizationParametersAddMin(builder, min_offset)
                if max_offset:
                    tflite.QuantizationParametersAddMax(builder, max_offset)
                quant_offset = tflite.QuantizationParametersEnd(builder)

            tflite.TensorStart(builder)
            tflite.TensorAddShape(builder, shape_offset)
            tflite.TensorAddType(builder, new_type)
            tflite.TensorAddBuffer(builder, tensor.Buffer())
            if name_offset:
                tflite.TensorAddName(builder, name_offset)
            if quant_offset:
                tflite.TensorAddQuantization(builder, quant_offset)
            tensor_offset = tflite.TensorEnd(builder)
            tensor_offsets.append(tensor_offset)

        # Operators
        op_offsets = []
        for op_i in range(sg.OperatorsLength()):
            op = sg.Operators(op_i)

            in_list = [op.Inputs(j) for j in range(op.InputsLength())]
            out_list = [op.Outputs(j) for j in range(op.OutputsLength())]

            tflite.OperatorStartInputsVector(builder, len(in_list))
            for x in reversed(in_list):
                builder.PrependInt32(x)
            inputs_offset = builder.EndVector(len(in_list))

            tflite.OperatorStartOutputsVector(builder, len(out_list))
            for x in reversed(out_list):
                builder.PrependInt32(x)
            outputs_offset = builder.EndVector(len(out_list))

            builtin_options_type = op.BuiltinOptionsType()
            builtin_options_offset = copy_builtin_options(builder, op)

            tflite.OperatorStart(builder)
            tflite.OperatorAddOpcodeIndex(builder, op.OpcodeIndex())
            tflite.OperatorAddInputs(builder, inputs_offset)
            tflite.OperatorAddOutputs(builder, outputs_offset)
            tflite.OperatorAddBuiltinOptionsType(builder, builtin_options_type)
            if builtin_options_offset:
                tflite.OperatorAddBuiltinOptions(builder, builtin_options_offset)
            op_offsets.append(tflite.OperatorEnd(builder))

        # SubGraph inputs/outputs
        subg_in_list = [sg.Inputs(i) for i in range(sg.InputsLength())]
        tflite.SubGraphStartInputsVector(builder, len(subg_in_list))
        for x in reversed(subg_in_list):
            builder.PrependInt32(x)
        inputs_offset = builder.EndVector(len(subg_in_list))

        subg_out_list = [sg.Outputs(i) for i in range(sg.OutputsLength())]
        tflite.SubGraphStartOutputsVector(builder, len(subg_out_list))
        for x in reversed(subg_out_list):
            builder.PrependInt32(x)
        outputs_offset = builder.EndVector(len(subg_out_list))

        # Build Tensors vector
        tflite.SubGraphStartTensorsVector(builder, len(tensor_offsets))
        for toff in reversed(tensor_offsets):
            builder.PrependUOffsetTRelative(toff)
        tensors_vector = builder.EndVector(len(tensor_offsets))

        # Build Operators vector
        tflite.SubGraphStartOperatorsVector(builder, len(op_offsets))
        for ooff in reversed(op_offsets):
            builder.PrependUOffsetTRelative(ooff)
        ops_vector = builder.EndVector(len(op_offsets))

        # SubGraph name
        name_offset = 0
        if sg.Name():
            name_offset = builder.CreateString(sg.Name().decode("utf-8"))

        tflite.SubGraphStart(builder)
        tflite.SubGraphAddTensors(builder, tensors_vector)
        tflite.SubGraphAddOperators(builder, ops_vector)
        tflite.SubGraphAddInputs(builder, inputs_offset)
        tflite.SubGraphAddOutputs(builder, outputs_offset)
        if name_offset:
            tflite.SubGraphAddName(builder, name_offset)
        subgraph_offset = tflite.SubGraphEnd(builder)
        subgraph_offsets.append(subgraph_offset)

    # Build subgraphs vector
    tflite.ModelStartSubgraphsVector(builder, len(subgraph_offsets))
    for sg_off in reversed(subgraph_offsets):
        builder.PrependUOffsetTRelative(sg_off)
    subgraphs_vec = builder.EndVector(len(subgraph_offsets))

    # Build operator codes vector
    tflite.ModelStartOperatorCodesVector(builder, len(op_code_offsets))
    for oc_off in reversed(op_code_offsets):
        builder.PrependUOffsetTRelative(oc_off)
    opcodes_vec = builder.EndVector(len(op_code_offsets))

    # Build buffers vector
    tflite.ModelStartBuffersVector(builder, len(buffer_offsets))
    for b_off in reversed(buffer_offsets):
        builder.PrependUOffsetTRelative(b_off)
    buffers_vec = builder.EndVector(len(buffer_offsets))

    # Create the new model
    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, old_model.Version())
    tflite.ModelAddSubgraphs(builder, subgraphs_vec)
    tflite.ModelAddOperatorCodes(builder, opcodes_vec)
    tflite.ModelAddBuffers(builder, buffers_vec)
    # If old_model has metadata, copy it similarly.
    final_model_offset = tflite.ModelEnd(builder)

    builder.Finish(final_model_offset)
    new_data = builder.Output()

    return new_data

def convert_tflite_float32_to_float16(input_tflite: str, output_tflite: str):
    """
    Reads `input_tflite`, converts float32 tensors -> float16,
    updates quantized_bias_type from FLOAT32->FLOAT16,
    and writes the modified model to `output_tflite`.
    """

    # Read old file
    with open(input_tflite, "rb") as f:
        old_data = f.read()

    new_data = convert_flatbuffer_float32_to_float16(old_data)

    # Write out
    with open(output_tflite, "wb") as f:
        f.write(new_data)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <input_model.tflite> <output_model.tflite>")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    convert_tflite_float32_to_float16(in_file, out_file)
