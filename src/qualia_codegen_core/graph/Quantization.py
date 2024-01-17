from __future__ import annotations

from dataclasses import dataclass

from qualia_codegen_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .RoundMode import RoundMode  # noqa: TCH001

@dataclass
class Quantization:
    number_type: type[int | float] | None = None
    width: int | None = None
    long_width: int | None = None
    weights_scale_factor: int | None = None
    bias_scale_factor: int | None = None
    output_scale_factor: int | None = None
    weights_round_mode: RoundMode | None = None
    output_round_mode: RoundMode | None = None
