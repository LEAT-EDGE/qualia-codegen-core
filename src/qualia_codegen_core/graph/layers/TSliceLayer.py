from dataclasses import dataclass

from .TBaseLayer import TBaseLayer


@dataclass
class TSliceLayer(TBaseLayer):
    slices: tuple[slice, ...]
