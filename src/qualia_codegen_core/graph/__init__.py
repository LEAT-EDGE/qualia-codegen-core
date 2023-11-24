import importlib.util
import logging

from .ModelGraph import ModelGraph
from .Quantization import Quantization

logger = logging.getLogger(__name__)

if importlib.util.find_spec('tensorflow') is None:
    logger.warning('Cannot find TensorFlow, Keras framework will be unavailable')
elif importlib.util.find_spec('keras') is None:
    logger.warning('Cannot find Keras, Keras framework will be unavailable')
else:
    from .KerasModelGraph import KerasModelGraph

if importlib.util.find_spec('torch') is None:
    logger.warning('Cannot find PyTorch, PyTorch framework will be unavailable')
else:
    from .TorchModelGraph import TorchModelGraph

__all__ = ['KerasModelGraph', 'ModelGraph', 'Quantization', 'TorchModelGraph']
