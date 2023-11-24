from dataclasses import dataclass

from .TActivationLayer import TActivation
from .TBaseLayer import TBaseLayer


@dataclass
class TAddLayer(TBaseLayer):
    activation: TActivation = TActivation.LINEAR
