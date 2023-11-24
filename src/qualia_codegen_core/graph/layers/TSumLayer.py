from dataclasses import dataclass

from .TBaseLayer import TBaseLayer


@dataclass
class TSumLayer(TBaseLayer):
    dim: tuple[int, ...]
