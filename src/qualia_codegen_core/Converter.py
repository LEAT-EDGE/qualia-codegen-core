# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

from __future__ import annotations

import logging
import sys
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import jinja2

from .Allocator import Allocator
from .DataConverter import DataConverter
from .graph import layers
from .graph.layers.TActivationLayer import TActivation, TActivationLayer
from .Quantizer import Quantizer
from .Validator import Validator

if TYPE_CHECKING:
    from .graph.LayerNode import LayerNode
    from .graph.layers.TBaseLayer import TBaseLayer
    from .graph.ModelGraph import ModelGraph

logger = logging.getLogger(__name__)

class NumberType(NamedTuple):
    number_type: type[int | float]
    width: int
    long_width: int
    min_val: int
    max_val: int

class Converter:
    layer_template_files: ClassVar[dict[type[TBaseLayer], str | None]] = {
        # Standard layers
        layers.TAvgPooling1DLayer: 'averagepool1d',
        layers.TAvgPooling2DLayer: 'averagepool2d',
        layers.TConv1DLayer: 'conv1d',
        layers.TConv2DLayer: 'conv2d',
        layers.TDenseLayer: 'fc',
        layers.TMaxPooling1DLayer: 'maxpool1d',
        layers.TMaxPooling2DLayer: 'maxpool2d',
        layers.TActivationLayer: 'activation',
        layers.TFlattenLayer: 'flatten',
        layers.TBatchNormalization1DLayer: 'batchnorm1d',
        layers.TBatchNormalization2DLayer: 'batchnorm2d',
        layers.TInputLayer: None,  # Nothing to generate for input layer

        # Custom Qualia layers
        layers.TAddLayer: 'add',
        layers.TSumLayer: 'sum', # Global Sum Pooling
    }

    TEMPLATE_PATH = files('qualia_codegen_core.assets')

    def __init__(self, output_path: Path | None = None) -> None:
        super().__init__()

        self.validator = Validator()
        self.dataconverter = DataConverter()

        if output_path:
            self.output_path = output_path
            self.output_path_header = output_path / 'include'
            self.output_path_weights = output_path / 'weights'

            self.output_path.mkdir(parents=True, exist_ok=True)
            self.output_path_header.mkdir(parents=True, exist_ok=True)
            self.output_path_weights.mkdir(parents=True, exist_ok=True)

            self.write_file = True
        else:
            self.output_path = Path()
            self.output_path_header = Path()
            self.output_path_weights = Path()

            self.write_file = False

        self.number_types = {NumberType(int, 32, 64, -(2 ** (32 - 1)), 2 ** (32 - 1) - 1)}

        self._template_path: list[Path] | None = None
        if isinstance(Converter.TEMPLATE_PATH, Path): # Already Path objected, no need for hackery
            self._template_path = [Converter.TEMPLATE_PATH]
        elif sys.version_info >= (3, 10): # Python 3.10 may return MultiplexedPath
            from importlib.readers import MultiplexedPath
            if isinstance(Converter.TEMPLATE_PATH, MultiplexedPath):
                self._template_path = [Converter.TEMPLATE_PATH / ''] # / operator applies to underlying Path

    def weights2carray(self, node: LayerNode) -> dict[str, dict[str, str | tuple[int, ...]]]:
        return {name: self.dataconverter.tensor2carray(arr, f'{node.layer.name}_{name}')
                for name, arr in node.layer.weights.items()}

    def write_layer_function(self, template: str, node: LayerNode) -> str:
        return self.render_template('layers/' + template + '.cc',
                                    self.output_path / f'{node.layer.name}.c',
                                    node=node,
                                    qtype2ctype=self.dataconverter.qtype2ctype)

    def write_layer_header(self, template: str, node: LayerNode) -> str:
        return self.render_template('include/layers/' + template + '.hh',
                                    self.output_path_header / f'{node.layer.name}.h',
                                    node=node,
                                    qtype2ctype=self.dataconverter.qtype2ctype)

    def write_layer_weights(self, template: str, node: LayerNode) -> str:
        return self.render_template('layers/weights/' + template + '.cc',
                                    self.output_path_weights / f'{node.layer.name}.c',
                                    node=node,
                                    weights=self.weights2carray(node))

    def render_template(self,
                        name: str,
                        out: Path,
                        **kwargs: Any) -> str: # noqa: ANN401 We really want to be able to pass anything to the rendered template
        if self._template_path is None:
            return ''

        template = jinja2.Environment(loader=jinja2.FileSystemLoader(searchpath=self._template_path),
                                      autoescape=jinja2.select_autoescape()).get_template(name)

        rendered = template.render(**kwargs)
        if self.write_file:
            with out.open('w', encoding='utf-8') as f:
                _ = f.write(rendered)
        return rendered

    def write_model_header(self, modelgraph: ModelGraph) -> str:
        return self.render_template('include/model.hh', self.output_path_header / 'model.h', nodes=modelgraph.nodes,
                                    qtype2ctype=self.dataconverter.qtype2ctype)

    def write_model(self,
                    modelgraph: ModelGraph,
                    allocation: dict[str, list[list[LayerNode]] | dict[LayerNode, int]] | None) -> str:
        return self.render_template('model.cc', self.output_path / 'model.c', nodes=modelgraph.nodes,
                                    allocation=allocation,
                                    qtype2ctype=self.dataconverter.qtype2ctype)

    def write_numeric_header(self) -> str:
        return self.render_template('include/number.hh', self.output_path_header / 'number.h',
                                    number_types=self.number_types,
                                    qtype2ctype=self.dataconverter.qtype2ctype)

    def write_defines_header(self, modelgraph: ModelGraph) -> str:
        return self.render_template('include/defines.hh', self.output_path_header / 'defines.h', nodes=modelgraph.nodes)

    def combine_zeropadding(self, modelgraph: ModelGraph) -> ModelGraph | None:
        zeropaddingnodes = [node for node in modelgraph.nodes if isinstance(node.layer, layers.TZeroPaddingLayer)]
        for zeropaddingnode in zeropaddingnodes:
            for outnode in zeropaddingnode.outnodes:
                if not hasattr(outnode.layer, 'padding'):
                    logger.error('Cannot fuse pading: "%s" does not have a padding attribute', outnode.layer.name)
                    return None
                # Double check since set doesn't contain layer type
                if not isinstance(zeropaddingnode.layer, layers.TZeroPaddingLayer):
                    return None
                outnode.layer.padding = zeropaddingnode.layer.padding
                outnode.layer.input_shape = zeropaddingnode.layer.input_shape
            modelgraph.delete_node(zeropaddingnode)
        return modelgraph

    def remove_dropout(self, modelgraph: ModelGraph) -> ModelGraph:
        dropoutnodes = [node for node in modelgraph.nodes if isinstance(node.layer, layers.TDropoutLayer)]
        for dropoutnode in dropoutnodes:
            modelgraph.delete_node(dropoutnode)
        return modelgraph

    def combine_relu(self, modelgraph: ModelGraph) -> ModelGraph | None:
        relunodes = [node for node in modelgraph.nodes
                        if isinstance(node.layer, layers.TActivationLayer)
                        and node.layer.activation in [TActivation.RELU, TActivation.RELU6]]
        for relunode in relunodes:
            for innode in relunode.innodes:  # warning: activations_range unsupported with multiple inputs to relu
                if not hasattr(innode.layer, 'activation'):
                    logger.error('Cannot fuse activation: "%s" does not have an activation attribute', innode.layer.name)
                    return None
                innode.layer.activation = cast(TActivationLayer, relunode.layer).activation
                innode.q.output_scale_factor = relunode.q.output_scale_factor
            modelgraph.delete_node(relunode)
        return modelgraph

    def remove_identity(self, modelgraph: ModelGraph) -> ModelGraph:
        identitynodes = [node for node in modelgraph.nodes if isinstance(node.layer, layers.TIdentityLayer)]
        for identitynode in identitynodes:
            modelgraph.delete_node(identitynode)
        return modelgraph

    # Operators (Add…) layers have names invalid as C tokens
    def rename_operators(self, modelgraph: ModelGraph) -> ModelGraph:
        for node in modelgraph.nodes:
            node.layer.name = node.layer.name.replace('.', '')
        return modelgraph

    def optimize_modelgraph(self, modelgraph: ModelGraph) -> ModelGraph | None:
        # Remove Indentity layers, useless
        modelgraph_no_identity = self.remove_identity(modelgraph)
        # Remove Dropout layers, useless during inference
        modelgraph_no_dropout = self.remove_dropout(modelgraph_no_identity)
        # Combine ZeroPadding with next layer (Conv1D)
        modelgraph_combined_zeropadding = self.combine_zeropadding(modelgraph_no_dropout)
        if modelgraph_combined_zeropadding is None:
            return None
        # Combine ReLU with previous layer (Conv1D/Dense), activations range must be copied to previous layer
        return self.combine_relu(modelgraph_combined_zeropadding)

    def preprocess_modelgraph(self, modelgraph: ModelGraph) -> ModelGraph | None:
        logger.info('ModelGraph:\n%s', modelgraph)

        optimized_modelgraph = self.optimize_modelgraph(modelgraph)
        if optimized_modelgraph is None:
            logger.error('Could not optimize ModelGraph')
            return None

        logger.info('ModelGraph after optimization:\n%s', optimized_modelgraph)
        graphviz = optimized_modelgraph.graphviz()
        if graphviz:
            logger.info('Graphviz: %s', graphviz)

        # Rename operator layers that are not valid identifiers for C
        return self.rename_operators(optimized_modelgraph)

    def validate_modelgraph(self, modelgraph: ModelGraph) -> bool:
        return all(self.validator.validate_node(node) for node in modelgraph.nodes)

    def quantize_modelgraph(self, modelgraph: ModelGraph) -> bool:
        for node in modelgraph.nodes:
            if node.q.number_type is None or node.q.width is None or node.q.long_width is None:
                logger.error('Missing quantization information for "%s"', node.layer.name)
                return False

            # Apply weights quantization for each layer with fixed point and weights
            if node.q.number_type == int and hasattr(node.layer, 'weights'):
                quantizer = Quantizer(width=node.q.width)
                if not quantizer.quantize_weights(node):
                    logger.error('Weights quantization failed for "%s"', node.layer.name)
                    return False

            # Add type layer in type list
            t = NumberType(node.q.number_type,
                 node.q.width,
                 node.q.long_width,
                 -(2 ** (node.q.width - 1)),
                 2 ** (node.q.width - 1) - 1)
            self.number_types.add(t)
        return True

    def generate_code(self, modelgraph: ModelGraph,
                      allocation: dict[str, list[list[LayerNode]] | dict[LayerNode, int]]) -> str | None:
        # Used to ignore includes in generated files for combined returned code
        rendered = '#define SINGLE_FILE\n'

        # Write defines.h global defines
        rendered += self.write_defines_header(modelgraph)

        # Write number.h numeric type configuration
        rendered += self.write_numeric_header()

        for node in modelgraph.nodes:
            template = self.layer_template_files[node.layer.__class__]
            # Skip layers with no code to generate
            if template is None:
                continue

            rendered += self.write_layer_header(template=template, node=node) + '\n'
            rendered += self.write_layer_function(template=template, node=node) + '\n'
            if hasattr(node.layer, 'weights') and len(node.layer.weights) > 0:
                rendered += self.write_layer_weights(template=template, node=node) + '\n'


        rendered += self.write_model_header(modelgraph=modelgraph) + '\n'
        rendered += self.write_model(modelgraph=modelgraph, allocation=allocation) + '\n'

        return rendered


    def convert_model(self, modelgraph: ModelGraph) -> str | None:
        if self._template_path is None:
            logger.error('Could not discover template path from module')
            return None

        final_modelgraph = self.preprocess_modelgraph(modelgraph)
        if final_modelgraph is None:
            logger.error('Could not preprocess ModelGraph')
            return None

        if not self.validate_modelgraph(final_modelgraph):
            logger.error('ModelGraph validation failed')
            return None

        if not self.quantize_modelgraph(final_modelgraph):
            return None

        allocator = Allocator()
        allocation = allocator(modelgraph)
        if not allocation:
            logger.error('Allocation failed')
            return None

        return self.generate_code(final_modelgraph, allocation)
