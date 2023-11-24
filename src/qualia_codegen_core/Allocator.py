# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .graph.layers import TFlattenLayer

if TYPE_CHECKING:
    import sys
    if sys.version_info >= (3, 11):
        from typing import Self
    else:
        from typing_extensions import Self

    from .graph import ModelGraph
    from .graph.LayerNode import LayerNode

logger = logging.getLogger(__name__)

class Allocator:
    @dataclass
    class AllocInfo:
        node: LayerNode
        input_ai: list[Self]
        keep_until: int
        overwrite_input: bool

    def __call__(self, modelgraph: ModelGraph) -> dict[str, list[list[LayerNode]] | dict[LayerNode, int]] | None:
        pools: list[list[Allocator.AllocInfo]] = [[]]

        alloc_info_list: list[Allocator.AllocInfo] = []

        for node in modelgraph.nodes[:-1]:  # No allocation for input and last layer, allocated by caller
            overwrite_input = isinstance(node.layer, TFlattenLayer)

            # First layer is assumed to take input from outside model
            inlayersi = [alloc_info_list[modelgraph.nodes.index(innode)] for innode in node.innodes]

            outlayersi = [modelgraph.nodes.index(outnode) for outnode in node.outnodes]
            keep_until = max(outlayersi)

            alloc_info_list.append(Allocator.AllocInfo(
                node,
                inlayersi,
                keep_until,
                overwrite_input))

        for i, a in enumerate(alloc_info_list[1:]):  # Skip InputLayer
            if i == 0:  # first layer after input layer, assume it takes input from outside model
                pools[0].append(a)
            elif a.overwrite_input:
                if len(a.input_ai) != 1:
                    logger.error('Need exactly one inpurt layer when overwriting input')
                    return None
                # Find which pool contains input
                inp = [p for p in pools if a.input_ai[0] in p]
                if len(inp) != 1:
                    logger.error('Input layer must be allocated in exactly one pool')
                    return None
                inp[0].append(a)
            else:
                # Find pools not containing inputs
                ap = [p for p in pools for iai in a.input_ai if iai not in p]
                # Find pools that can be overwritten
                op = [p for p in ap if p[-1].keep_until <= i]

                if len(op) < 1:  # no free pool, allocate new one
                    pools.append([a])
                else:  # Add to first usable pool — maybe possible to optimize allocation size
                    op[0].append(a)

        return {
            'pools': [[a.node for a in p] for p in pools],
            'index': {a.node: (i + 1) for i, p in enumerate(pools) for a in p},
        }
