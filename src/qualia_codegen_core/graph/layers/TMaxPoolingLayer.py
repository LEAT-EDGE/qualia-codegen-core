from dataclasses import dataclass

from .TActivationLayer import TActivation
from .TBaseLayer import TBaseLayer


@dataclass
class TMaxPoolingLayer(TBaseLayer):
    activation: TActivation
    pool_size: tuple[int, ...]
    strides: tuple[int, ...]
