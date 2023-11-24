from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quantization:
    number_type: type[int | float] | None = None
    width: int | None = None
    long_width: int | None = None
    weights_scale_factor: int | None = None
    output_scale_factor: int | None = None
