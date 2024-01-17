from __future__ import annotations

import logging

import numpy as np

from qualia_codegen_core.graph.RoundMode import RoundMode
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
                                         scale_factor: int,
                                         round_mode: RoundMode) -> NDArrayFloatOrInt | None:
        target_dtype = getattr(np, f'int{self.width}', None)
        if target_dtype is None: # Initialization failed due to unsupported width
            logger.error('No integer data type for width %s', self.width)
            return None

        # Already integer, no need to quantize
        if np.issubdtype(arr.dtype, np.integer):
            return arr

        new_arr = arr * (1 << scale_factor)
        if round_mode == RoundMode.FLOOR:
            new_arr = np.floor(new_arr)
        elif round_mode == RoundMode.NEAREST:
            new_arr = np.floor(new_arr + 0.5)
        else:
            logger.error('Unsupported round mode: %s, supported: floor, nearest', round_mode)
            raise ValueError

        new_arr = np.clip(new_arr, self.number_min, self.number_max)
        return new_arr.astype(target_dtype)

    def quantize_weights_with_scale_factor(self,  # noqa: PLR0913
                                           node: LayerNode,
                                           scale_factor: int,
                                           round_mode: RoundMode,
                                           bias_scale_factor: int | None = None,
                                           exclude: list[str] | None = None) -> bool:

        for weights_name, weights in node.layer.weights.items():
            # Skip excluded weights
            if exclude and weights_name in exclude:
                continue

            if bias_scale_factor is not None and weights_name == 'bias': # Quantize biases with their own scale factor if it exists
                new_weights = self.quantize_array_with_scale_factor(weights, scale_factor=bias_scale_factor, round_mode=round_mode)
            else:
                new_weights = self.quantize_array_with_scale_factor(weights, scale_factor=scale_factor, round_mode=round_mode)
            if new_weights is None:
                return False


            setattr(node.layer, weights_name, new_weights)

        return True

    def quantize_weights(self, node: LayerNode, exclude: list[str] | None = None) -> bool:
        if len(node.layer.weights) > 0:
            if node.q.weights_scale_factor is None:
                logger.error('No weights quantization information for %s', node.layer.name)
                return False

            if node.q.weights_round_mode is None:
                logger.error('No round mode select for %s', node.layer.name)
                return False

            logger.info('%s quantization weights=%s', node.layer.name, node.q.weights_scale_factor)
            return self.quantize_weights_with_scale_factor(node,
                                                           node.q.weights_scale_factor,
                                                           round_mode=node.q.weights_round_mode,
                                                           bias_scale_factor=node.q.bias_scale_factor,
                                                           exclude=exclude)
        return True
