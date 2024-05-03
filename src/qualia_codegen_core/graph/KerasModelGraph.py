from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, cast

from keras.activations import linear, relu, softmax  # type: ignore[import-untyped] # No stubs for keras package
from keras.layers import (  # type: ignore[import-untyped] # No stubs for keras package
    Activation,
    Add,
    AveragePooling1D,
    AveragePooling2D,
    BatchNormalization,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Layer,
    MaxPooling1D,
    MaxPooling2D,
    ZeroPadding1D,
    ZeroPadding2D,
)

from qualia_codegen_core.typing import DTypes, NDArrayFloatOrInt, Shape, ShapeOptional, Shapes

from .layers import (
    TActivationLayer,
    TAddLayer,
    TAvgPooling1DLayer,
    TAvgPooling2DLayer,
    TBaseLayer,
    TBatchNormalization1DLayer,
    TBatchNormalization2DLayer,
    TConv1DLayer,
    TConv2DLayer,
    TDenseLayer,
    TDropoutLayer,
    TFlattenLayer,
    TInputLayer,
    TMaxPooling1DLayer,
    TMaxPooling2DLayer,
    TZeroPadding1DLayer,
    TZeroPadding2DLayer,
)
from .layers.TActivationLayer import TActivation
from .ModelGraph import ModelGraph

try:
    # Keras 3.x
    from keras.layers import InputLayer
except ImportError:
    try:
        # Keras >= 2.13.1
        from keras.src.engine.input_layer import InputLayer  # type: ignore[import-untyped] # No stubs for keras package
    except ImportError:
        # Keras < 2.13.0
        from keras.engine.input_layer import InputLayer  # type: ignore[import-untyped] # No stubs for keras package

if importlib.util.find_spec('keras.src.ops.node'): # Keras 3.x
    from keras.src.ops.node import Node  # type: ignore[import-untyped]
else:
    from keras.src.engine.node import Node  # type: ignore[import-untyped]

if TYPE_CHECKING:
    import numpy.typing
    import tensorflow as tf  # type: ignore[import-untyped]
    from keras import Model  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class KerasModelGraph(ModelGraph):
    MAPPING: ClassVar[dict[type[Layer], Callable[[Layer], tuple[type[TBaseLayer], list[Any]]]]] = {
        InputLayer: lambda _: (TInputLayer, []),
        ZeroPadding1D: lambda layer: (TZeroPadding1DLayer, [layer.padding]),
        ZeroPadding2D: lambda layer: (TZeroPadding2DLayer, [layer.padding]),
        BatchNormalization: lambda layer: (TBatchNormalization1DLayer if len(KerasModelGraph.__get_input_shape(layer)) == 3  # noqa: PLR2004
                                           else TBatchNormalization2DLayer,
                                           [TActivation.LINEAR,
                                            layer.moving_mean.numpy(),
                                            layer.moving_variance.numpy(),
                                            layer.gamma.numpy(),
                                            layer.beta.numpy(),
                                            layer.epsilon]),
        Conv1D: lambda layer: (TConv1DLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation],
                                              KerasModelGraph.__transpose(layer.kernel.numpy()),
                                              layer.kernel_size,
                                              layer.strides,
                                              layer.filters,
                                              layer.use_bias,
                                              layer.bias.numpy(),
                                              layer.groups,
                                              layer.padding]),
        Conv2D: lambda layer: (TConv2DLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation],
                                              KerasModelGraph.__transpose(layer.kernel.numpy()),
                                              layer.kernel_size,
                                              layer.strides,
                                              layer.filters,
                                              layer.use_bias,
                                              layer.bias.numpy(),
                                              layer.groups,
                                              layer.padding]),
        Dropout: lambda layer: (TDropoutLayer, [layer.rate]),
        MaxPooling1D: lambda layer: (TMaxPooling1DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        MaxPooling2D: lambda layer: (TMaxPooling2DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        AveragePooling1D: lambda layer: (TAvgPooling1DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        AveragePooling2D: lambda layer: (TAvgPooling2DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        Activation: lambda layer: (TActivationLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation]]),
        Add: lambda _: (TAddLayer, []),
        Flatten: lambda _: (TFlattenLayer, []),
        Dense: lambda layer: (TDenseLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation],
                                            KerasModelGraph.__transpose(layer.kernel.numpy()),
                                            layer.units,
                                            layer.use_bias,
                                            layer.bias.numpy()]),
    }

    ACTIVATION_MAPPING: ClassVar[dict[Callable[[tf.Tensor], tf.Tensor], TActivation]] = {
        relu: TActivation.RELU,
        softmax: TActivation.SOFTMAX,
        linear: TActivation.LINEAR,
    }

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.__model = model
        self.__layer_cache: dict[Layer, TBaseLayer] = {}

    def convert(self) -> ModelGraph | None:
        for layer in self.__model.layers:
            conv = self.__convert(layer)
            if conv is False: # Conversion failure
                return None
            inlayers = self.__get_layer_input_layers(layer)
            if inlayers is False:
                return None # Conversion failure
            self.add_layer(conv, inlayers=inlayers)
        return self

    def __none_to_one_shape(self, shape: ShapeOptional) -> Shape:
        return Shape(s if s is not None else 1 for s in shape)

    def __convert_shapes(self, shapes: tuple[int | None, ...] | list[tuple[int | None, ...]]) -> Shapes:
        """Convert shape or list of shapes to Shapes object.

        Some layers (e.g. InputLayer) may have an unneeded list of shape.
        """
        if isinstance(shapes, list):
            return Shapes(Shape(self.__none_to_one_shape(ShapeOptional(s))) for s in shapes)
        return Shapes((Shape(self.__none_to_one_shape(ShapeOptional(shapes))),))

    def __convert_dtypes(self, dtypes: numpy.typing.DTypeLike | list[numpy.typing.DTypeLike]) -> DTypes:
        """Convert dtype or list of dtypes to DTypes object."""
        if isinstance(dtypes, list):
            return DTypes(t for t in dtypes)
        return DTypes((dtypes,))

    def __convert(self, layer: Layer) -> Literal[False] | TBaseLayer:
        existing = self.__layer_cache.get(layer, None)
        if existing is not None:
            return existing

        args = [self.__convert_shapes(self.__get_input_shape(layer)),
                self.__convert_shapes(self.__get_output_shape(layer)),
                self.__convert_dtypes(layer.dtype),
                layer.name]

        options = KerasModelGraph.MAPPING.get(type(layer), None)

        if options is None:
            logger.error('Unsupported Keras layer type: %s', type(layer))
            return False

        cls, params = options(layer)

        self.__layer_cache[layer] = res = cls(*args, *params)
        return res

    def __get_layer_input_layers(self, layer: Layer) -> Literal[False] | list[Layer]:
        inlayers = []
        for n in self.__get_inbound_nodes(layer):
            # Keras 3.x compatibility
            if hasattr(n, 'operation'):
                inlayer = self.__convert(n.operation)
                if inlayer is False:
                    return False
                inlayers.append(inlayer)
            else:
                for inb in n.iterate_inbound():
                    inlayer = self.__convert(inb[0])
                    if inlayer is False:
                        return False
                    inlayers.append(inlayer)
        return inlayers

    @classmethod
    def __transpose(cls, weights: NDArrayFloatOrInt) -> NDArrayFloatOrInt:
        # Shape requirement for CMSIS-NN, also used without CMSIS-NN for compatibility
        if len(weights.shape) == 4: # noqa: PLR2004
            return weights.transpose(3, 0, 1, 2)
        if len(weights.shape) == 3: # noqa: PLR2004
            return weights.transpose(2, 0, 1)
        if len(weights.shape) == 2: # noqa: PLR2004
            return weights.swapaxes(0, 1)
        raise NotImplementedError

    @classmethod
    def __get_input_shape(cls, layer: Layer) -> tuple[int, ...]:
        # Keras 3.x compatibility
        if hasattr(layer, 'get_build_config'):
            if isinstance(layer, InputLayer): # get_build_config() returns None for InputLayer
                return cast(tuple[int, ...], layer.batch_shape)
            return cast(tuple[int, ...], layer.get_build_config()['input_shape'])
        return cast(tuple[int, ...], layer.input_shape)

    @classmethod
    def __get_output_shape(cls, layer: Layer) -> tuple[int, ...]:
        # Keras 3.x compatibility
        if hasattr(layer, 'compute_output_shape'):
            if isinstance(layer, InputLayer): # compute_output_shape not implemented for InputLayer
                return cast(tuple[int, ...], layer.batch_shape)

            return cast(tuple[int, ...], layer.compute_output_shape(cls.__get_input_shape(layer)))

        return cast(tuple[int, ...], layer.output_shape)

    @classmethod
    def __get_inbound_nodes(cls, layer: Layer) -> list[Node]:
        if hasattr(layer, 'inbound_nodes'):
            return cast(list[Node], layer.inbound_nodes)
        # Keras 3.x compatibility, no public API
        return [n for inbound_node in layer._inbound_nodes for n in inbound_node.parent_nodes]  # noqa: SLF001
