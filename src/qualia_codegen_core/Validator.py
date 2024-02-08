# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from qualia_codegen_core.graph.RoundMode import RoundMode
from qualia_codegen_core.typing import TYPE_CHECKING

from .graph.layers import TActivationLayer, TBaseLayer, TBatchNormalizationLayer, TFlattenLayer, TSumLayer
from .graph.layers.TActivationLayer import TActivation

if TYPE_CHECKING:
    import sys
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard
    from qualia_codegen_core.graph.LayerNode import LayerNode  # noqa: TCH001

logger = logging.getLogger(__name__)

@dataclass
class TBaseLayerWithActivation(TBaseLayer):
    activation: TActivation

class Validator:
    """Class used to validate various parts of the model to make sure they conform to the limitation of the conversion tool."""

    def __has_activation_attribute(self, layer: TBaseLayer) -> TypeGuard[TActivationLayer | TBaseLayerWithActivation]:
        return hasattr(layer, 'activation')

    def validate_combined_activation(self, layer: TBaseLayer) -> bool:
        if self.__has_activation_attribute(layer):
            if layer.activation == TActivation.SOFTMAX and not isinstance(layer, TActivationLayer):  # not a standalone softmax
                    logger.error('Softmax activation must be used as a standalone layer, not combined to another layer (%s)',
                                 layer.__class__.__name__)
                    return False

            if layer.activation not in [TActivation.LINEAR, TActivation.RELU, TActivation.RELU6]:
                logger.error('Activation function %s not supported', layer.activation)
                return False

        # No activation is ok as well
        return True

    def validate_batchnorm(self, node: LayerNode) -> bool:
        if not isinstance(node.layer, TBatchNormalizationLayer):  # Not BatchNorm, ignore
            return True

        # Detect variable depth not matching input_shape channels
        wrong_shape_vars = {var: getattr(node.layer, var) for var in ['kernel', 'bias'] if
                            getattr(node.layer, var).shape != node.input_shape[0][-1:]}
        for name, var in wrong_shape_vars.items():
            logger.error('Variable %s with shape %s does not match last dimension of input shape %s',
                         name, var.shape, node.input_shape[-1:])

        return not wrong_shape_vars

    def validate_flatten(self, node: LayerNode) -> bool:
        if not isinstance(node.layer, TFlattenLayer):  # Not Flatten, ignore
            return True

        if len(node.innodes) != 1:
            logger.error('Flatten should only have one input node')
            return False

        if len(node.output_shape) != 1:
            logger.error('Flatten should only have one output shape')
            return False

        inelements = math.prod(math.prod(e[1:]) for e in node.input_shape)

        if inelements != node.output_shape[0][-1]:
            logger.error('Number of elements for Flatten input_shape (%s) and output_shape (%s) do not match',
                         inelements, node.output_shape[-1])
            return False

        return True

    def validate_global_sum_pooling(self, node: LayerNode) -> bool:
        if isinstance(node.layer, TSumLayer):
            if node.outnodes:
                logger.error('Sum is only supported as the last layer for Global Sum Pooling')
                return False
            if len(node.innodes) != 1:
                logger.error('Global Sum Pooling should only have one input node')
                return False

            if len(node.input_shape[0]) == 3: # noqa: PLR2004 1D, (N, S, C)
                if node.layer.dim != (-1,):
                    logger.error('Global Sum Pooling 1D should apply to last dimension (-1), dim=%s', node.layer.dim)
                    return False
            elif len(node.input_shape[0]) == 4: # noqa: PLR2004 2D, (N, H, W, C)
                if node.layer.dim != (-2, -1):
                    logger.error('Global Sum Pooling 2D should apply to the two last dimensions (-2, -1), dim=%s', node.layer.dim)
                    return False
            else:
                logger.error('Global Sum Pooling input should have 3 (1D: N, S, C) or 4 (2D: N, H, W, C) dimensions')
                return False
        return True

    def validate_round_mode(self, node: LayerNode) -> bool:
        """Check if layer activation round mode is not None when number_type is int.

        :param node: LayerNode to check the activation round mode of
        :return: ``True`` if layer's round mode is neither None nor :attr:`qualia_codegen_core.graph.RoundMode.RoundMode.NONE` when
                 layer's number_type is int, otherwise ``False``
        """
        if isinstance(node.q.number_type, int) and (
                node.q.output_round_mode is None or node.q.output_round_mode == RoundMode.NONE):
            logger.error('Round mode must not be None when number type is int for layer %s', node.layer.name)
            return False
        return True


    def validate_node(self, node: LayerNode) -> bool:
        valid = True

        # Validate combined activation
        valid = valid and self.validate_combined_activation(node.layer)
        valid = valid and self.validate_batchnorm(node)
        valid = valid and self.validate_flatten(node)
        valid = valid and self.validate_global_sum_pooling(node)
        return valid and self.validate_round_mode(node)
