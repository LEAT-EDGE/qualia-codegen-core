from __future__ import annotations

import logging

import numpy as np

from qualia_codegen_core.typing import TYPE_CHECKING, NDArrayFloatOrInt

if TYPE_CHECKING:
    from .graph.LayerNode import LayerNode  # noqa: TCH001

logger = logging.getLogger(__name__)

class Quantizer:

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.number_min = -(2 ** (width - 1))
        self.number_max = 2 ** (width - 1) - 1

    def quantize_array_with_scale_factor(self,
                                         arr: NDArrayFloatOrInt,
                                         scale_factor: int) -> NDArrayFloatOrInt | None:
        target_dtype = getattr(np, f'int{self.width}', None)
        if target_dtype is None: # Initialization failed due to unsupported width
            logger.error('No integer data type for width %s', self.width)
            return None

        # Already integer, no need to quantize
        if np.issubdtype(arr.dtype, np.integer):
            return arr

        new_arr = arr * (1 << scale_factor)
        new_arr = np.floor(new_arr)
        new_arr = np.clip(new_arr, self.number_min, self.number_max)
        return new_arr.astype(target_dtype)

    def quantize_weights_with_scale_factor(self, node: LayerNode, scale_factor: int, exclude: list[str] | None = None) -> bool:

        for weights_name, weights in node.layer.weights.items():
            # Skip excluded weights
            if exclude and weights_name in exclude:
                continue

            new_weights = self.quantize_array_with_scale_factor(weights, scale_factor=scale_factor)
            if new_weights is None:
                return False

            setattr(node.layer, weights_name, new_weights)

        return True

    def quantize_weights(self, node: LayerNode, exclude: list[str] | None = None) -> bool:
        if len(node.layer.weights) > 0:
            if node.q.weights_scale_factor is None:
                logger.error('No weights quantization information for %s', node.layer.name)
                return False
            logger.info('%s quantization weights=%s', node.layer.name, node.q.weights_scale_factor)
            return self.quantize_weights_with_scale_factor(node, node.q.weights_scale_factor, exclude=exclude)
        return True
