from dataclasses import dataclass

from .TBaseLayer import TBaseLayer


@dataclass
class TPermuteLayer(TBaseLayer):
    dims: tuple[int, ...]
