from dataclasses import dataclass

from .TBaseLayer import TBaseLayer


@dataclass
class TDropoutLayer(TBaseLayer):
    p: float
