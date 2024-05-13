from dataclasses import dataclass

from .TConvLayer import TConvLayer


@dataclass
class TConv1DLayer(TConvLayer):
    padding: tuple[int, int]
