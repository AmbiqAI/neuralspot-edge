import keras


def glu(dim: int = -1, hard: bool = False, name: str | None = None) -> keras.Layer:
    """Gated linear unit layer"""

    def layer(x: keras.KerasTensor) -> keras.KerasTensor:
        out, gate = keras.ops.split(x, indices_or_sections=2, axis=dim)
        act = keras.activations.sigmoid if hard else keras.activations.hard_sigmoid
        gate = keras.layers.Activation(act)(gate)
        x = keras.layers.Multiply()([out, gate])
        return x

    # END DEF

    return layer


def relu(name: str | None = None, truncated: bool = False) -> keras.Layer:
    """ReLU activation layer"""
    name = name + ".act" if name else None
    act = keras.activations.relu6 if truncated else keras.activations.relu
    return keras.layers.Activation(activation=act, name=name)


def swish(name: str | None = None, hard: bool = False) -> keras.Layer:
    """Swish activation layer"""
    name = name + ".act" if name else None
    act = keras.activations.hard_swish if hard else keras.activations.swish
    return keras.layers.Activation(act, name=name)


def relu6(name: str | None = None) -> keras.Layer:
    """Hard ReLU activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation("relu6", name=name)


def mish(name: str | None = None) -> keras.Layer:
    """Mish activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation(keras.activations.mish, name=name)


def gelu(name: str | None = None) -> keras.Layer:
    """GeLU activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation(keras.activations.gelu, name=name)


def sigmoid(name: str | None = None, hard: bool = False) -> keras.Layer:
    """Sigmoid activation layer"""
    name = name + ".act" if name else None
    activation = keras.activations.hard_sigmoid if hard else keras.activations.sigmoid
    return keras.layers.Activation(activation, name=name)


def hard_sigmoid(name: str | None = None) -> keras.Layer:
    """Hard sigmoid activation layer"""
    name = name + ".act" if name else None
    return keras.layers.Activation(keras.activations.hard_sigmoid, name=name)
