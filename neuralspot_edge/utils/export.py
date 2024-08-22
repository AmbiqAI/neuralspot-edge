"""
# NSE Keras Registration API

This module provides a decorator to register Keras models and layers for serialization.

Classes:
    nse_export: Decorator to register Keras models and layers for serialization

"""

from types import FunctionType

import keras

# Disable namex for now
namex = None
# try:
#     import namex
# except ImportError:
#     namex = None


def maybe_register_serializable(symbol, package):
    if isinstance(symbol, FunctionType) or hasattr(symbol, "get_config"):
        keras.saving.register_keras_serializable(package=package)(symbol)


if namex:

    class nse_export(namex.export):
        def __init__(self, path=None, package="nse"):
            if path is None:
                path = f"neuralspot_edge.{package}"
            super().__init__(package="nse", path=path)
            self.package = package

        def __call__(self, symbol):
            maybe_register_serializable(symbol, self.package)
            return super().__call__(symbol)

else:

    class nse_export:
        def __init__(self, path=None, package="nse"):
            if path is None:
                path = f"neuralspot_edge.{package}"
            self.package = package

        def __call__(self, symbol):
            maybe_register_serializable(symbol, self.package)
            return symbol
