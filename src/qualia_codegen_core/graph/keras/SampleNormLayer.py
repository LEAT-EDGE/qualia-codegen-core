import logging
import sys

import keras
import tensorflow as tf

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

logger = logging.getLogger(__name__)

class SampleNormLayer(keras.layers.Layer):
    """custom layer to normalize input data."""

    def __init__(self,
                 norm: str = 'z',
                 name: str = '',
                 trainable: bool = False,  # noqa: FBT001, FBT002
                 dtype: str = 'float32') -> None:
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        if norm not in ['z', 'minmax']:
            logger.error('Unsupported mode %s, supported modes: z, minmax', norm)
            raise ValueError
        self.norm = norm

    @override
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Normalize each input sample."""
        if self.norm == 'z':
            inputs -= tf.math.reduce_mean(inputs, axis=1, keepdims=True)
            inputs /= tf.math.reduce_std(inputs, axis=1, keepdims=True)
        elif self.norm == 'minmax':
            inputs -= tf.math.reduce_min(inputs, axis=1, keepdims=True)
            inputs /= tf.math.reduce_max(inputs, axis=1, keepdims=True)
        return inputs
