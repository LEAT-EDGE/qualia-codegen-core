from dataclasses import dataclass

from .TConvLayer import TConvLayer


@dataclass
class TConv2DLayer(TConvLayer):
    padding: tuple[tuple[int, int], tuple[int, int]]
