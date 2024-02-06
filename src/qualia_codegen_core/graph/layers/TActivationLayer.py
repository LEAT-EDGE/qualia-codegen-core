from dataclasses import dataclass
from enum import Enum

from .TBaseLayer import TBaseLayer


class TActivation(Enum):
    RELU = 0
    RELU6 = 1
    SOFTMAX = 2
    LINEAR = 3
    IF = 4


@dataclass
class TActivationLayer(TBaseLayer):
    activation: TActivation
