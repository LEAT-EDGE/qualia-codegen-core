# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.
# April 29, 2021

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Iterable
from itertools import zip_longest
from typing import TYPE_CHECKING, Callable, cast

from .LayerNode import LayerNode
from .layers import TBaseLayer, TInputLayer

if TYPE_CHECKING:
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

    from torch import nn # noqa: I001 # torch must be imported before keras to avoid deadlock
    import keras.Model # type: ignore[import-untyped] # No stubs for keras package

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class ModelGraph:
    def __init__(self, nodes: list[LayerNode] | None = None) -> None:
        super().__init__()
        self.__nodes = nodes or []

    def add_node(self,
                 node: LayerNode,
                 innodes: Iterable[LayerNode] | None = None,
                 outnodes: Iterable[LayerNode] | None = None) -> None:
        innodes = innodes or []
        outnodes = outnodes or []

        node.innodes.extend(
            innode for innode in innodes if innode not in node.innodes)  # could  be nicer to use a set but we need to keep order
        node.outnodes.extend(outnode for outnode in outnodes if outnode not in node.outnodes)

        for innode in innodes:
            if node not in innode.outnodes:
                innode.outnodes.append(node)
        for outnode in outnodes:
            if node not in outnode.innodes:
                outnode.innodes.append(node)

        if node in self.__nodes:
            logger.warning('Node already exists in graph')

        self.__nodes.append(node)

    def delete_node(self, node: LayerNode) -> None:
        for innode in node.innodes:
            # Disconnect layer to remove from output of previous layer
            index = innode.outnodes.index(node)
            _ = innode.outnodes.pop(index)
            # Connect outputs from layer to remove to output of previous layer
            # Try to preserve insertion location and ordering
            for i, e in enumerate(node.outnodes):
                innode.outnodes.insert(index + i, e)
        for outnode in node.outnodes:
            # Disconnect layer to remove from input of next layer
            index = outnode.innodes.index(node)
            _ = outnode.innodes.pop(index)
            # Connect inputs from layer to remove to input of next layer
            # Try to preserve insertion location and ordering
            for i, e in enumerate(node.innodes):
                outnode.innodes.insert(index + i, e)
        self.__nodes.remove(node)  # Remove layer from list

    # Delete each node for which predicate function is true
    def delete_node_if(self, predicate: Callable[[LayerNode], bool]) -> None:
        to_delete = [n for n in self.nodes if predicate(n)]
        for n in to_delete:
            self.delete_node(n)

    def replace_node(self, oldnode: LayerNode, newnode: LayerNode) -> None:
        self.add_node(newnode, oldnode.innodes, oldnode.outnodes)
        self.delete_node(oldnode)

    def find_node_from_layer(self, layer: TBaseLayer) -> LayerNode | None:
        nodes = [node for node in self.nodes if node.layer is layer]
        if len(nodes) == 0:
            return None
        if len(nodes) > 1:
            logger.warning('More than one node found for layer, returning the first one')
        return nodes[0]

    def get_nodes_for_layers(self, layers: TBaseLayer | Iterable[TBaseLayer]) -> tuple[LayerNode | None, ...]:
        if isinstance(layers, Iterable):
            return tuple(n for layer in layers for n in self.get_nodes_for_layers(layer))
        return (self.find_node_from_layer(layers), )

    def no_none_in_nodes(self, nodes: Iterable[LayerNode | None]) -> TypeGuard[Iterable[LayerNode]]:
      return all(node is not None for node in nodes)

    def add_layer(self,
                  layer: TBaseLayer,
                  inlayers: list[TBaseLayer] | None = None,
                  outlayers: list[TBaseLayer] | None = None) -> None:
        if self.find_node_from_layer(layer):
            return

        inlayers = inlayers or []
        outlayers = outlayers or []

        # Special case for missing InputLayer in case of Sequential model
        for inlayer in inlayers:
            # If InputLayer does not exist in graph
            if isinstance(inlayer, TInputLayer) and inlayer not in [n.layer for n in self.nodes]:
                self.add_layer(inlayer)  # No input, linking to output is handled by the next add_node

        innodes = self.get_nodes_for_layers(inlayers)
        if not self.no_none_in_nodes(innodes):
            logger.error('Input node for layer %s not found', layer.name)
            return
        outnodes = self.get_nodes_for_layers(outlayers)
        if not self.no_none_in_nodes(outnodes):
            logger.error('Output node for layer %s not found', layer.name)
            return

        self.add_node(LayerNode(layer), innodes, outnodes)

    @property
    def nodes(self) -> list[LayerNode]:
        return self.__nodes

    @override
    def __str__(self) -> str:
        pad = 48

        header = f'{"Inputs": <{pad}} | {"Layer": <{pad}} | {"Outputs": <{pad}} | {"Input shape": <{pad}} | {"Output shape": <{pad}}\n' # noqa: E501
        s = '—' * len(header) + '\n'
        s += header
        s += '—' * len(header) + '\n'
        for node in self.nodes:
            for inlayername, layername, outlayername, inshape, outshape in zip_longest(
                    [n.layer.name for n in node.innodes],
                    [node.layer.name],
                    [n.layer.name for n in node.outnodes],
                    [str(s) for s in node.layer.input_shape],
                    [str(node.layer.output_shape)], fillvalue=''):
                s += f'{inlayername: <{pad}} | {layername: <{pad}} | {outlayername: <{pad}} | {inshape: <{pad}} | {outshape: <{pad}}\n' # noqa: E501
            s += '-' * len(header) + '\n'
        return s

    def graphviz(self) -> str | None:
        try:
            from graphviz import Digraph  # type: ignore[import-untyped] # Graphviz is missing py.typed xflr6/graphviz#180
        except ImportError:
            logger.warning('Graphviz not available')
            return None

        grph = Digraph()
        for node in self.nodes:
            for out in node.outnodes:
                grph.edge(node.layer.name, out.layer.name)
        return cast(str, grph.source)

    @classmethod
    def auto_detect(cls, obj: keras.Model | nn.Module) -> ModelGraph:
        if importlib.util.find_spec('torch') is not None:
            from torch import nn

            if isinstance(obj, nn.Module):
                from .TorchModelGraph import TorchModelGraph
                return TorchModelGraph(obj)

        from .KerasModelGraph import KerasModelGraph
        return KerasModelGraph(obj)
