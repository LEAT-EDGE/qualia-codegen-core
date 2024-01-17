from __future__ import annotations

from dataclasses import dataclass

from qualia_codegen_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qualia_codegen_core.graph.RoundMode import RoundMode  # noqa: TCH001

@dataclass
class ActivationRange:
    input_q: int | None
    activation_q: int | None
    weights_q: int | None
    bias_q: int | None
    input_round_mode: RoundMode | None
    activation_round_mode: RoundMode | None
    weights_round_mode: RoundMode | None
