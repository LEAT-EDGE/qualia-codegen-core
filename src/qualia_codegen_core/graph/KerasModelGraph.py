from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, cast

from keras.activations import linear, relu, softmax  # type: ignore[import-untyped] # No stubs for keras package
from keras.layers import (  # type: ignore[import-untyped] # No stubs for keras package
    Activation,
    Add,
    AveragePooling1D,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
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

from .keras.SampleNormLayer import SampleNormLayer
from .layers import (
    TActivationLayer,
    TAddLayer,
    TAvgPooling1DLayer,
    TAvgPooling2DLayer,
    TBaseLayer,
    TBatchNormalization1DLayer,
    TBatchNormalization2DLayer,
    TConcatenateLayer,
    TConv1DLayer,
    TConv2DLayer,
    TDenseLayer,
    TDropoutLayer,
    TFlattenLayer,
    TInputLayer,
    TMaxPooling1DLayer,
    TMaxPooling2DLayer,
    TSampleNormLayer,
    TSliceLayer,
    TZeroPadding1DLayer,
    TZeroPadding2DLayer,
)
from .layers.TActivationLayer import TActivation
from .layers.TSampleNormLayer import TSampleNormMode
from .ModelGraph import ModelGraph

try:
    # Keras 3.x
    from keras.layers import InputLayer
    from keras.src.ops.node import Node  # type: ignore[import-untyped]
    from keras.src.ops.numpy import GetItem  # type: ignore[import-untyped]
except ImportError:
    try:
        # Keras >= 2.13.1
        from keras.src.engine.input_layer import InputLayer  # type: ignore[import-untyped] # No stubs for keras package
    except ImportError:
        # Keras < 2.13.0
        from keras.engine.input_layer import InputLayer  # type: ignore[import-untyped] # No stubs for keras package
    from keras.src.engine.node import Node  # type: ignore[import-untyped]
    from keras.src.layers.core.tf_op_layer import SlicingOpLambda  # type: ignore[import-untyped] # No stubs for keras package

if TYPE_CHECKING:
    import numpy.typing
    import tensorflow as tf
    from keras import Model  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class KerasModelGraph(ModelGraph):
    MAPPING: ClassVar[dict[type[Layer], Callable[[Layer, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]]] = {
        InputLayer: lambda *_: (TInputLayer, []),
        ZeroPadding1D: lambda layer, _: (TZeroPadding1DLayer, [layer.padding]),
        ZeroPadding2D: lambda layer, _: (TZeroPadding2DLayer, [layer.padding]),
        BatchNormalization: lambda layer, _: (TBatchNormalization1DLayer if len(KerasModelGraph.__get_input_shape(layer)) == 3  # noqa: PLR2004
                                           else TBatchNormalization2DLayer,
                                           [TActivation.LINEAR,
                                            layer.moving_mean.numpy(),
                                            layer.moving_variance.numpy(),
                                            layer.gamma.numpy(),
                                            layer.beta.numpy(),
                                            layer.epsilon]),
        Conv1D: lambda layer, args: (TConv1DLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation],
                                              KerasModelGraph.__transpose(layer.kernel.numpy()),
                                              layer.kernel_size,
                                              layer.strides,
                                              layer.filters,
                                              layer.use_bias,
                                              layer.bias.numpy(),
                                              layer.groups,
                                              KerasModelGraph.__compute_padding1d(layer.padding,
                                                                                  layer.kernel_size,
                                                                                  layer.strides,
                                                                                  args.input_shape)]),
        Conv2D: lambda layer, args: (TConv2DLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation],
                                              KerasModelGraph.__transpose(layer.kernel.numpy()),
                                              layer.kernel_size,
                                              layer.strides,
                                              layer.filters,
                                              layer.use_bias,
                                              layer.bias.numpy(),
                                              layer.groups,
                                              KerasModelGraph.__compute_padding2d(layer.padding,
                                                                                  layer.kernel_size,
                                                                                  layer.strides,
                                                                                  args.input_shape)]),
        Dropout: lambda layer, _: (TDropoutLayer, [layer.rate]),
        MaxPooling1D: lambda layer, _: (TMaxPooling1DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        MaxPooling2D: lambda layer, _: (TMaxPooling2DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        AveragePooling1D: lambda layer, _: (TAvgPooling1DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        AveragePooling2D: lambda layer, _: (TAvgPooling2DLayer, [TActivation.LINEAR, layer.pool_size, layer.strides]),
        Activation: lambda layer, _: (TActivationLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation]]),
        Add: lambda *_: (TAddLayer, []),
        Flatten: lambda *_: (TFlattenLayer, []),
        Dense: lambda layer, _: (TDenseLayer, [KerasModelGraph.ACTIVATION_MAPPING[layer.activation],
                                            KerasModelGraph.__transpose(layer.kernel.numpy()),
                                            layer.units,
                                            layer.use_bias,
                                            layer.bias.numpy()]),

        # BrainMIX layer
        SampleNormLayer: lambda layer, _: (TSampleNormLayer, [KerasModelGraph.SAMPLENORM_MODE_MAPPING[layer.norm]]),
        Concatenate: lambda *_: (TConcatenateLayer, []),
    }

    try: # Keras 3.x
        MAPPING[GetItem] = lambda layer, _: (TSliceLayer, [layer._inbound_nodes[0].arguments.args[1]])
    except NameError: # Keras 2.x
        MAPPING[SlicingOpLambda] = lambda layer, _: (TSliceLayer,
                                                     [tuple(slice(s['start'], s['stop'], s['step'])
                                                            for s in layer.inbound_nodes[0].call_kwargs['slice_spec'])])

    ACTIVATION_MAPPING: ClassVar[dict[Callable[[tf.Tensor], tf.Tensor], TActivation]] = {
        relu: TActivation.RELU,
        softmax: TActivation.SOFTMAX,
        linear: TActivation.LINEAR,
    }

    SAMPLENORM_MODE_MAPPING: ClassVar[dict[str, TSampleNormMode]] = {
        'z': TSampleNormMode.ZSCORE,
        'minmax': TSampleNormMode.MINMAX,
    }

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.__model = model
        self.__layer_cache: dict[Layer, TBaseLayer] = {}

    def convert(self) -> ModelGraph | None:
        for layer in self.__model.layers:
            if not self.__convert_and_add_layer(layer):
                return None
        return self

    def __convert_and_add_layer(self, layer: Layer) -> bool:
        conv = self.__convert(layer)
        if conv is False: # Conversion failure
            return False

        inlayers = self.__get_layer_input_layers(layer)

        converted_inlayers: list[TBaseLayer] = []
        for inlayer in inlayers:
            converted_inlayer = self.__convert(inlayer)
            if converted_inlayer is False:
                return False # Conversion failure

            # In case the Keras layers list is not ordered properly, some input layers may not have been added to our graph yet
            if not self.find_node_from_layer(converted_inlayer) and not self.__convert_and_add_layer(inlayer):
                return False # Conversion failure

            converted_inlayers.append(converted_inlayer)

        self.add_layer(conv, inlayers=converted_inlayers)
        return True

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

        # Not the final layer type, just used to collect the common args
        args = TBaseLayer(input_shape=self.__convert_shapes(self.__get_input_shape(layer)),
                output_shape=self.__convert_shapes(self.__get_output_shape(layer)),
                output_dtype=self.__convert_dtypes(layer.dtype if hasattr(layer, 'dtype') else layer.output.dtype),
                name=layer.name)

        options = KerasModelGraph.MAPPING.get(type(layer), None)

        if options is None:
            logger.error('Unsupported Keras layer type: %s', type(layer))
            return False

        cls, params = options(layer, args)

        self.__layer_cache[layer] = res = cls(*dataclasses.astuple(args), *params)
        return res

    def __get_layer_input_layers(self, layer: Layer) -> list[Layer]:
        inlayers: list[TBaseLayer] = []
        for n in self.__get_inbound_nodes(layer):
            # Keras 3.x compatibility
            if hasattr(n, 'operation'):
                inlayers.append(n.operation)
            else:
                for inb in n.iterate_inbound():
                    inlayers.append(inb[0])
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
    def __compute_padding1d(cls,
                            padding: str,
                            kernel_size: tuple[int],
                            strides: tuple[int],
                            input_shape: Shapes) -> tuple[int, int]:
        if padding == 'valid':
            return (0, 0)

        if padding == 'same':
            # Inspired from tensorflow/core/framework/kernel_shape_util.cc:GetWindowedOutputSizeVerbose()
            output_size = (input_shape[0][-2] + strides[0] - 1) // strides[0]
            padding_needed = max(0, (output_size - 1) * strides[0] + kernel_size[0] - input_shape[0][-2])
            padding_before = padding_needed // 2
            padding_after = padding_needed - padding_before
            return (padding_before, padding_after)

        logger.error('Unsupported padding mode %s', padding)
        raise ValueError

    @classmethod
    def __compute_padding2d(cls,
                            padding: str,
                            kernel_size: tuple[int, int],
                            strides: tuple[int, int],
                            input_shape: Shapes) -> tuple[tuple[int, int], tuple[int, int]]:
        if padding == 'valid':
            return ((0, 0), (0, 0))

        if padding == 'same':
            # Inspired from tensorflow/core/framework/kernel_shape_util.cc:GetWindowedOutputSizeVerbose()
            output_size_x = (input_shape[0][-2] + strides[1] - 1) // strides[1]
            output_size_y = (input_shape[0][-3] + strides[0] - 1) // strides[0]
            padding_x_needed = max(0, (output_size_x - 1) * strides[1] + kernel_size[1] - input_shape[0][-2])
            padding_y_needed = max(0, (output_size_y - 1) * strides[0] + kernel_size[0] - input_shape[0][-3])
            padding_x_before = padding_x_needed // 2
            padding_y_before = padding_y_needed // 2
            padding_x_after = padding_x_needed - padding_x_before
            padding_y_after = padding_y_needed - padding_y_before
            return ((padding_y_before, padding_y_after), (padding_x_before, padding_x_after))

        logger.error('Unsupported padding mode %s', padding)
        raise ValueError

    @classmethod
    def __get_input_shape(cls, layer: Layer) -> tuple[int, ...]:
        # Keras 3.x compatibility
        if not hasattr(layer, 'input_shape'):
            if isinstance(layer, InputLayer): # get_build_config() returns None for InputLayer
                return cast(tuple[int, ...], layer.batch_shape)
            if hasattr(layer, 'get_build_config'):
                return cast(tuple[int, ...], layer.get_build_config()['input_shape'])
            # Some operations do not have get_build_config() so try to use input tensor shape instead
            return cast(tuple[int, ...], layer.input.shape)
        return cast(tuple[int, ...], layer.input_shape)

    @classmethod
    def __get_output_shape(cls, layer: Layer) -> tuple[int, ...]:
        # Keras 3.x compatibility
        if not hasattr(layer, 'output_shape'):
            if isinstance(layer, InputLayer): # compute_output_shape not implemented for InputLayer
                return cast(tuple[int, ...], layer.batch_shape)
            if hasattr(layer, 'compute_output_shape'):
                return cast(tuple[int, ...], layer.compute_output_shape(cls.__get_input_shape(layer)))
            # Some operations do not have compute_output_shape() so try to use input tensor shape instead
            return cast(tuple[int, ...], layer.output.shape)
        return cast(tuple[int, ...], layer.output_shape)

    @classmethod
    def __get_inbound_nodes(cls, layer: Layer) -> list[Node]:
        if hasattr(layer, 'inbound_nodes'):
            return cast(list[Node], layer.inbound_nodes)
        # Keras 3.x compatibility, no public API
        return [n for inbound_node in layer._inbound_nodes for n in inbound_node.parent_nodes]  # noqa: SLF001
