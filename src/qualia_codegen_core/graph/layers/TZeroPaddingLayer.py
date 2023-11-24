from __future__ import annotations

from dataclasses import dataclass

from .TBaseLayer import TBaseLayer


@dataclass
class TZeroPaddingLayer(TBaseLayer):
    padding: int | tuple[int, int]
