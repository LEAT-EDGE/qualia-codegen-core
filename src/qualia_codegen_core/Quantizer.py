from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .graph.LayerNode import LayerNode

logger = logging.getLogger(__name__)

class Quantizer:

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.number_min = -(2 ** (width - 1))
        self.number_max = 2 ** (width - 1) - 1

    def quantize_weights_with_scale_factor(self, node: LayerNode, scale_factor: int) -> bool:
        target_dtype = getattr(np, f'int{self.width}', None)
        if target_dtype is None: # Initialization failed due to unsupported width
            logger.error('No integer data type for width %s', self.width)
            return False

        for weights_name, weights in node.layer.weights.items():
            # Already integer, no need to quantize
            if np.issubdtype(weights.dtype, np.integer):
                continue

            new_weights = weights * (1 << scale_factor)
            new_weights = np.floor(new_weights)
            new_weights = np.clip(new_weights, self.number_min, self.number_max)
            new_weights = new_weights.astype(target_dtype)

            setattr(node.layer, weights_name, new_weights)
        return True

    def quantize_weights(self, node: LayerNode) -> bool:
        if len(node.layer.weights) > 0:
            if node.q.weights_scale_factor is None:
                logger.error('No weights quantization information for %s', node.layer.name)
                return False
            logger.info('%s quantization weights=%s', node.layer.name, node.q.weights_scale_factor)
            return self.quantize_weights_with_scale_factor(node, node.q.weights_scale_factor)
        return True
