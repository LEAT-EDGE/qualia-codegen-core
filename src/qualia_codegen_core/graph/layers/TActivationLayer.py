from dataclasses import dataclass, field
from enum import Enum

from .TBaseLayer import TBaseLayer


class TActivation(Enum):
    RELU = 0
    SOFTMAX = 1
    LINEAR = 2
    IF = 3


@dataclass
class TActivationLayer(TBaseLayer):
    activation: TActivation
    bias: object = field(default_factory=list)
    use_bias: bool = False
