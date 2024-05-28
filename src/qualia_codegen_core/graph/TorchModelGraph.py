from __future__ import annotations

import dataclasses
import functools
import logging
import operator
import sys
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal, Union, cast

import numpy as np
import numpy.typing
import torch
from torch import Tensor, fx
from torch.fx._symbolic_trace import Tracer
from torch.fx.node import Argument, Node
from torch.nn import (
    AdaptiveAvgPool1d,
    AvgPool1d,
    AvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Dropout,
    Flatten,
    Identity,
    Linear,
    MaxPool1d,
    MaxPool2d,
    Module,
    ReLU,
    ReLU6,
)
from torch.nn.functional import adaptive_avg_pool2d

from qualia_codegen_core.typing import DTypes, NDArrayFloatOrInt, Shape, Shapes

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
    TIdentityLayer,
    TInputLayer,
    TMaxPooling1DLayer,
    TMaxPooling2DLayer,
    TSumLayer,
)
from .layers.TActivationLayer import TActivation
from .ModelGraph import ModelGraph

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

IterableNode = Union[Node, Iterable['IterableNode']]

logger = logging.getLogger(__name__)

class TorchModelGraph(ModelGraph):
    MODULE_MAPPING: ClassVar[dict[type[Module], Callable[[Module, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]]] = {
        # Standard torch.nn layers
        BatchNorm1d: lambda module, _: (TBatchNormalization1DLayer, [TActivation.LINEAR,
                                                                     module.running_mean.detach().numpy()
                                                                     if isinstance(module, BatchNorm1d)
                                                                     and module.running_mean is not None
                                                                     else None,
                                                                     module.running_var.detach().numpy()
                                                                     if isinstance(module, BatchNorm1d)
                                                                     and module.running_var is not None
                                                                     else None,
                                                                     cast(BatchNorm1d, module).weight.detach().numpy(),
                                                                     cast(BatchNorm1d, module).bias.detach().numpy(),
                                                                     cast(BatchNorm1d, module).eps]),
        BatchNorm2d: lambda module, _: (TBatchNormalization2DLayer, [TActivation.LINEAR,
                                                                     module.running_mean.detach().numpy()
                                                                     if isinstance(module, BatchNorm2d)
                                                                     and module.running_mean is not None
                                                                     else None,
                                                                     module.running_var.detach().numpy()
                                                                     if isinstance(module, BatchNorm2d)
                                                                     and module.running_var is not None
                                                                     else None,
                                                                     cast(BatchNorm2d, module).weight.detach().numpy(),
                                                                     cast(BatchNorm2d, module).bias.detach().numpy(),
                                                                     cast(BatchNorm2d, module).eps]),
        Conv1d: lambda module, _: (TConv1DLayer, [TActivation.LINEAR,
                                                  TorchModelGraph.transpose(cast(Conv1d, module).weight.detach().numpy()),
                                                  cast(Conv1d, module).kernel_size,
                                                  cast(Conv1d, module).stride,
                                                  cast(Conv1d, module).out_channels,
                                                  cast(Conv1d, module).bias is not None,
                                                  module.bias.detach().numpy()
                                                      if isinstance(module, Conv1d) and module.bias is not None else None,
                                                  cast(Conv1d, module).groups,
                                                  list(cast(Conv1d, module).padding) * 2]),
        Conv2d: lambda module, _: (TConv2DLayer, [TActivation.LINEAR,
                                                  TorchModelGraph.transpose(cast(Conv2d, module).weight.detach().numpy()),
                                                  cast(Conv2d, module).kernel_size,
                                                  cast(Conv2d, module).stride,
                                                  cast(Conv2d, module).out_channels,
                                                  cast(Conv2d, module).bias is not None,
                                                  module.bias.detach().numpy()
                                                      if isinstance(module, Conv2d) and module.bias is not None else None,
                                                  cast(Conv2d, module).groups,
                                                  ((cast(Conv2d, module).padding[0], ) * 2,
                                                   (cast(Conv2d, module).padding[1], ) * 2)]),
        ReLU: lambda *_: (TActivationLayer, [TActivation.RELU]),
        ReLU6: lambda *_: (TActivationLayer, [TActivation.RELU6]),
        MaxPool1d: lambda module, _: (TMaxPooling1DLayer, [TActivation.LINEAR,
                                                           TorchModelGraph.array_or_scalar(cast(MaxPool1d, module).kernel_size),
                                                           TorchModelGraph.array_or_scalar(cast(MaxPool1d, module).stride)]),
        MaxPool2d: lambda module, _: (TMaxPooling2DLayer, [TActivation.LINEAR,
                                                           TorchModelGraph.array_or_scalar(cast(MaxPool2d, module).kernel_size),
                                                           TorchModelGraph.array_or_scalar(cast(MaxPool2d, module).stride)]),
        AvgPool1d: lambda module, _: (TAvgPooling1DLayer, [TActivation.LINEAR,
                                                           cast(AvgPool1d, module).kernel_size,
                                                           cast(AvgPool1d, module).stride]),
        AvgPool2d: lambda module, _: (TAvgPooling2DLayer, [TActivation.LINEAR,
                                                           cast(AvgPool2d, module).kernel_size,
                                                           cast(AvgPool2d, module).stride]),
        AdaptiveAvgPool1d: lambda module, args: (TAvgPooling1DLayer,
                                                 [TActivation.LINEAR,
                                                  (args.input_shape[0][-1] // TorchModelGraph.array_or_scalar(
                                                     cast(AdaptiveAvgPool1d, module).output_size)[0], ),
                                                  (args.input_shape[0][-1] // TorchModelGraph.array_or_scalar(
                                                      cast(AdaptiveAvgPool1d, module).output_size)[0], )]),
        Flatten: lambda *_: (TFlattenLayer, []),
        Linear: lambda module, _: (TDenseLayer, [TActivation.LINEAR,
                                              cast(Linear, module).weight.detach().numpy(),
                                              cast(Linear, module).out_features, True,
                                              cast(Linear, module).bias.detach().numpy()]),
        Dropout: lambda module, _: (TDropoutLayer, [cast(Dropout, module).p]),
        Identity: lambda *_: (TIdentityLayer, []),
    }

    METHOD_MAPPING: ClassVar[dict[str, Callable[..., tuple[type[TBaseLayer], list[Any]]]]] = {
        'sum': lambda dim: (TSumLayer, [TorchModelGraph.array_or_scalar(dim)]),
    }

    FUNCTION_MAPPING: ClassVar[dict[Callable[..., Any], Callable[..., tuple[type[TBaseLayer], list[Any]]]]] = {
        operator.add: lambda *_: (TAddLayer, []),
        adaptive_avg_pool2d: lambda x, output_size: (TAvgPooling2DLayer,
                                                 [TActivation.LINEAR,
                                                  (x.shape[-2] // TorchModelGraph.array_or_scalar(
                                                     output_size)[0],
                                                   x.shape[-1] // TorchModelGraph.array_or_scalar(
                                                     output_size)[1]),
                                                  (x.shape[-2] // TorchModelGraph.array_or_scalar(
                                                      output_size)[0],
                                                   x.shape[-1] // TorchModelGraph.array_or_scalar(
                                                      output_size)[1])]),
       torch.flatten: lambda *_: (TFlattenLayer, []),
    }

    FUNCTION_INPUT_ARG_INDEX: ClassVar[dict[Callable[..., Any] | str, tuple[int, ...]]] = {
        operator.add: (0, 1),
        adaptive_avg_pool2d: (0,),
        torch.flatten: (0,),
    }

    # Custom tracer that generates call_module for our custom Qualia layers instead of attempting to trace their forward()
    class TracerCustomLayers(Tracer):
        def __init__(self, custom_layers: tuple[type[Module], ...]) -> None:
            super().__init__()
            self.custom_layers = custom_layers

        @override
        def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
            # Avoid tracing custom layers
            return super().is_leaf_module(m, module_qualified_name) or isinstance(m, self.custom_layers)

    def __init__(self, model: Module) -> None:
        super().__init__()


        self._model = model
        self.__layer_cache: dict[Node, TBaseLayer] = {}
        self.__layer_outputs: dict[str, Tensor] = {}
        self.__modules = dict(model.named_modules())
        self.__module_mapping: dict[type[Module], Callable[[Module, TBaseLayer], tuple[type[TBaseLayer], list[Any]]]] = {}

    def convert(self,
                custom_layers: dict[type[Module],
                                    Callable[[Module, TBaseLayer],
                                             tuple[type[TBaseLayer], list[Any]]]] | None = None) -> ModelGraph | None:
        custom_layers = custom_layers if custom_layers is not None else {}
        self.__module_mapping = {**TorchModelGraph.MODULE_MAPPING, **custom_layers}
        # Put model in inference mode
        _ = self._model.eval()

        tracer = TorchModelGraph.TracerCustomLayers(custom_layers=tuple(custom_layers.keys()))
        graph = tracer.trace(self._model)
        logger.info('Torch FX graph:')
        graph.print_tabular()

        for layer in graph.nodes:
            conv = self.__convert(layer)
            if conv is False:
                logger.error('Conversion failed')
                return None
            if conv is None: # End of graph
                break
            inlayers = self.__get_layer_input_layers(layer)
            if inlayers is False: # Conversion failure
                logger.error('Conversion failed')
                return None
            self.add_layer(conv, inlayers=inlayers)

        # channels_first to channels_last for first FC layer
        self.__reformat_fc_weights_data()

        return self

    def __shape_channels_last_to_first(self, shape: Shape) -> Shape:
        return Shape((shape[-1], ) + shape[0:-1])

    def __shape_channels_first_to_last(self, shape: Shape) -> Shape:
        return Shape(shape[1:] + (shape[0], ))

    @classmethod
    def transpose(cls, weights: NDArrayFloatOrInt) -> NDArrayFloatOrInt:
        # Shape requirement for CMSIS-NN, also used without CMSIS-NN for compatibility
        if len(weights.shape) == 3: # noqa: PLR2004
            return weights.transpose((0, 2, 1))
        if len(weights.shape) == 4: # noqa: PLR2004
            return weights.transpose((0, 2, 3, 1))
        raise NotImplementedError

    @classmethod
    def array_or_scalar(cls, obj: tuple[int, ...] | int) -> tuple[int, ...]:
        if isinstance(obj, Iterable):
            return tuple(obj)
        return (obj, )

    def __load_arg(self, a: Argument) -> Any: # noqa: ANN401 we have no knowledge about the arg types
        return fx.node.map_arg(a, lambda n: self.__layer_outputs[n.name])

    def __get_layer_output_shapes(self, args: IterableNode) -> Literal[False] | Shapes:
        def concat(x: Literal[False] | Shapes,
                   y: Literal[False] | Shapes) -> Literal[False] | Shapes:
            if x is False or y is False:
                return False
            return Shapes(x + y)

        if isinstance(args, Iterable):
            return functools.reduce(concat, map(self.__get_layer_output_shapes, args))

        layer = self.__convert(args)
        if layer is None or layer is False:
            logger.error('Invalid arg encountered while fetching input shapes: %s', args)
            return False
        return Shapes(layer.output_shape)

    def __get_layer_output_dtypes(self, args: IterableNode) -> Literal[False] | DTypes:
        def concat(x: Literal[False] | DTypes,
                   y: Literal[False] | DTypes) -> Literal[False] | DTypes:
            if x is False or y is False:
                return False
            return DTypes(x + y)

        if isinstance(args, Iterable):
            return functools.reduce(concat,
                                    map(self.__get_layer_output_dtypes, args))

        layer = self.__convert(args)
        if layer is None or layer is False:
            logger.error('Invalid arg encountered while fetching input dtypes: %s', args)
            return False
        return DTypes(layer.output_dtype)

    def __generate_dummy_inputs(self, shapes: Shapes,
                                dtypes: DTypes) -> tuple[Tensor, ...]:
        def generate_input(shape: Shape, dtype: numpy.typing.DTypeLike) -> Tensor:
            return torch.from_numpy(np.zeros((shape[0],
                                              *self.__shape_channels_last_to_first(Shape(shape[1:]))),
                                             dtype=dtype))
        return tuple(generate_input(shape, dtype) for shape, dtype in zip(shapes, dtypes))

    def __get_tensor_output_shapes(self, tensors: tuple[Tensor, ...]) -> Shapes:
        def get_shape(tensor: Tensor) -> Shape:
            return Shape((tensor.shape[0], *self.__shape_channels_first_to_last(Shape(tensor.shape[1:]))))
        return Shapes(get_shape(t) for t in tensors)


    def __get_tensor_output_dtypes(self, tensors: tuple[Tensor, ...]) -> Literal[False] | DTypes:
        def get_dtypes(tensor: Tensor) -> Literal[False] | numpy.typing.DTypeLike:
            x = tensor.detach().numpy()
            if isinstance(x, np.ndarray): # No typing from torch.Tensor.numpy()
                return cast(numpy.typing.DTypeLike, x.dtype)
            logger.error('Return type of tensor.detach().numpy() should be a numpy.ndarray')
            return False

        def no_false_value(it: Iterable[Literal[False] | numpy.typing.DTypeLike]) -> TypeGuard[Iterable[numpy.typing.DTypeLike]]:
            return all(t is not False for t in it)

        res = tuple(get_dtypes(t) for t in tensors)
        if no_false_value(res):
            return DTypes(res)
        return False

    def _convert_placeholder(self) -> TBaseLayer | None:
        if not hasattr(self._model, 'input_shape') or not isinstance(self._model.input_shape, tuple):
            logger.error('Model must have input_shape attribute')
            return None

        shp: Shape = Shape(self._model.input_shape)
        # Prepend dummy dimension
        shp = Shape((1, *shp))

        # Assume input is single-precision floating-point # WIP it could change
        return TInputLayer(Shapes((shp,)), Shapes((shp,)), DTypes((np.float32,)), 'input')

    def _convert_call_module(self, layer: Node) -> Literal[False] | TBaseLayer | None:
        module = self.__modules[layer.target]

        # Input arg to a Node should be a Node or Iterable of Node
        if not self.__is_iterablenode_recursive(layer.args):
            logger.error('Node arg type "%s" is not Node or Iterable of Node', type(layer.args))
            return False

        # Handle multiple inputs, for a possible Concat layer, Add layer handled as well even though shape should be identical
        inputs_shape = self.__get_layer_output_shapes(layer.args)
        if inputs_shape is False:
            logger.error('Could not get input shapes for "%s"', layer.target)
            return False

        inputs_dtype = self.__get_layer_output_dtypes(layer.args)
        if inputs_dtype is False:
            logger.error('Could not get input dtypes for "%s"', layer.target)
            return False

        dummy_inputs = self.__generate_dummy_inputs(inputs_shape, inputs_dtype)
        # Call module with original keyword arguments (for layers with parameters in forward())
        module_outputs = module(*dummy_inputs, **layer.kwargs)
        self.__layer_outputs[layer.name] = module_outputs
        dummy_outputs = (module_outputs,) if isinstance(module_outputs, Tensor) else tuple(module_outputs)

        outputs_shape = self.__get_tensor_output_shapes(dummy_outputs)

        outputs_dtype = self.__get_tensor_output_dtypes(dummy_outputs)
        if outputs_dtype is False:
            logger.error('Could not get output dtypes for "%s"', layer.target)
            return False

        # Not the final layer type, just used to collect the common args
        args = TBaseLayer(input_shape=inputs_shape,
                output_shape=outputs_shape,
                output_dtype=outputs_dtype,
                name=layer.name)

        options = self.__module_mapping.get(type(module), None)

        if options is None:
            logger.error('Unsupported Torch layer type: %s', type(module))
            return False

        layercls, opts = options(module, args)
        return layercls(*dataclasses.astuple(args), *opts) # args and opts must be specified in the correct order

    def _convert_call_method(self, layer: Node) -> TBaseLayer | None:
        if not isinstance(layer.target, str):
            logger.error('Method target type should be str, got: %s', type(layer.target))
            return None

        # self_obj is the object (layer's output tensor) the method is called on, args are the method's arguments
        self_obj, *method_args = self.__load_arg(layer.args)
        # kwargs are the methods's keyword arguments
        method_kwargs = self.__load_arg(layer.kwargs)
        method = getattr(self_obj, layer.target)
        module_outputs = method(*method_args, **method_kwargs)
        self.__layer_outputs[layer.name] = module_outputs
        dummy_outputs = (module_outputs,) if isinstance(module_outputs, Tensor) else tuple(module_outputs)

        if not hasattr(self_obj, 'shape') or not isinstance(self_obj.shape, tuple):
            logger.error('No or invalid input_shape found for %s', layer.target)
            return None

        inputs_shape = Shapes((Shape(self_obj.shape),))

        outputs_shape = self.__get_tensor_output_shapes(dummy_outputs)

        outputs_dtype = self.__get_tensor_output_dtypes(dummy_outputs)
        if outputs_dtype is False:
            logger.error('Could not get output dtypes for "%s"', layer.target)
            return None

        # Not the final layer type, just used to collect the common args
        args = TBaseLayer(input_shape=inputs_shape,
                output_shape=outputs_shape,
                output_dtype=outputs_dtype,
                name=layer.name)

        options = self.METHOD_MAPPING.get(layer.target, None)

        if options is None:
            logger.error('Unsupported Torch method: %s', layer.target)
            return None

        layercls, opts = options(*method_args, **method_kwargs)
        return layercls(*dataclasses.astuple(args), *opts) # args and opts must be specified in the correct order

    def _convert_call_function(self, layer: Node) -> TBaseLayer | None:
        if not callable(layer.target):
            logger.error('Function should be callable, got type: %s', type(layer.target))
            return None

        # self_obj is the object (layer's output tensor) the method is called on, args are the method's arguments
        function_args = self.__load_arg(layer.args)
        # kwargs are the methods's keyword arguments
        function_kwargs = self.__load_arg(layer.kwargs)
        function = layer.target
        function_outputs = function(*function_args, **function_kwargs)
        self.__layer_outputs[layer.name] = function_outputs
        dummy_outputs = (function_outputs,) if isinstance(function_outputs, Tensor) else tuple(function_outputs)

        # Extract function args which are Tensor (i.e., inputs) to get their shapes and filter out extra arguments
        # Assume result is of type Node which means a layer in the graph since that's what should generate Tensor.
        # Warning: not recursive, hopefully not a problem for a standalone function call
        args_tensor = [cast(Node, ref) for ref, val in zip(layer.args, function_args) if isinstance(val, Tensor)]

        # Handle multiple inputs, for a possible Concat layer, Add layer handled as well even though shape should be identical
        inputs_shape = self.__get_layer_output_shapes(args_tensor)
        if inputs_shape is False:
            logger.error('Could not get input shapes for "%s"', layer.target)
            return None

        outputs_shape = self.__get_tensor_output_shapes(dummy_outputs)

        outputs_dtype = self.__get_tensor_output_dtypes(dummy_outputs)
        if outputs_dtype is False:
            logger.error('Could not get output dtypes for "%s"', layer.target)
            return None

        # Not the final layer type, just used to collect the common args
        args = TBaseLayer(input_shape=inputs_shape,
                output_shape=outputs_shape,
                output_dtype=outputs_dtype,
                name=layer.name)

        options = self.FUNCTION_MAPPING.get(layer.target, None)

        if options is None:
            logger.error('Unsupported Torch function: %s', layer.target)
            return None

        layercls, opts = options(*function_args, **function_kwargs)
        return layercls(*dataclasses.astuple(args), *opts) # args and opts must be specified in the correct order

    def __convert_iterable(self, it: IterableNode) -> Literal[False] | tuple[TBaseLayer | None, ...]:
        convs: list[TBaseLayer | None] = []
        if isinstance(it, Iterable):
            for eit in it:
                convi = self.__convert_iterable(eit)
                if convi is False: # Conversion failed
                    return False
                convs += convi
        else:
            conv = self.__convert(it)
            if conv is False: # Conversion failed
                return False
            convs.append(conv)
        return tuple(convs)

    def __is_iterablenode_recursive(self, it: Argument) -> TypeGuard[IterableNode]:
        if isinstance(it, Iterable):
            return all(self.__is_iterablenode_recursive(e) for e in it)
        return isinstance(it, Node)

    def __convert(self, layer: Node) -> Literal[False] | TBaseLayer | None:
        existing = self.__layer_cache.get(layer, None)
        if existing is not None:
            return existing

        op = layer.op
        res: Literal[False] | TBaseLayer | None = None

        if op == 'output':
            return None # No layer to generate
        if op == 'placeholder':
            res = self._convert_placeholder()
        elif op == 'call_module':
            res = self._convert_call_module(layer)
            if res is None: # In case __convert_call_module ended up on 'output' op
                return None
        elif op == 'call_method':
            res = self._convert_call_method(layer)
        elif op == 'call_function':
            res = self._convert_call_function(layer)
        else: # Unsupported
            logger.error('Unsupported Torch op %s, type: %s', op, type(layer))
            return False

        if res is None or res is False: # Unsupported
            return False

        self.__layer_cache[layer] = res
        return res

    def __get_layer_input_layers(self, layer: Node) -> Literal[False] | list[TBaseLayer]:
        inlayers: list[TBaseLayer] = []
        args = layer.args

        # Special handling for functions where not all args should be considered as input, i.e., a node in the graph
        if layer.op == 'call_function':
            args = tuple(args[i] for i in self.FUNCTION_INPUT_ARG_INDEX[layer.target])

        for x in args:
            if not self.__is_iterablenode_recursive(x): # Input arg to a Node should be a Node or Iterable of Node
                logger.error('Node arg type "%s" is not Node or Iterable of Node', type(x))
                return False
            new_inlayers = self.__convert_iterable(x)
            if new_inlayers is False: # Conversion failure
                return False

            inlayers.extend([new_inlayer for new_inlayer in new_inlayers if new_inlayer is not None])
        return inlayers

    def __reformat_fc_weights_data(self) -> None:
        for node in self.nodes:
            if not isinstance(node.layer, TFlattenLayer):
                continue

            # After Flatten comes Dense, must reshape weights and swap axes for channels_last used by C code
            for outnode in node.outnodes:
                if not isinstance(outnode.layer, TDenseLayer):
                    continue

                dense = outnode.layer
                # reshape using Flatten input shape (for example last Conv output)
                dense.kernel = dense.kernel.reshape(
                        (dense.units, ) + node.layer.input_shape[0][-1:] + node.layer.input_shape[0][1:-1])
                dense.kernel = TorchModelGraph.transpose(dense.kernel)
                dense.kernel = dense.kernel.reshape((dense.units, -1))
