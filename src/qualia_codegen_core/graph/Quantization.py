"""Contains the Quantization class to hold quantization settings in a model graph node."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from qualia_codegen_core.typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .RoundMode import RoundMode


@dataclass
class Quantization:
    """Holds quantization settings in a model graph node."""

    #: Number abstract data type, either ``float`` for floating-point coding or ``int`` for fixed-point coding
    number_type: type[int | float] | None = None
    #: Total number of bits to use to encode a value (output, bias and weights)
    width: int | None = None
    #: Larger number of bits to use for temporary values during computation in C code
    long_width: int | None = None

    #: Not used in C code, total number of bits to use to encode weights
    weights_width: int | None = None
    #: Not used in C code, total number of bits to use to encode biases
    bias_width: int | None = None
    #: Not used in C code, total number of bits to use to encode outputs
    output_width: int | None = None

    #: Number of bits for the fractional part of weights
    weights_scale_factor: int | None = None
    #: Number of bits for the fractional part of biases
    bias_scale_factor: int | None = None
    #: Number of bits for the fractional part of outputs
    output_scale_factor: int | None = None

    #: Rounding mode to use when quantizing weights
    weights_round_mode: RoundMode | None = None
    #: Rounding mode to use when quantizing outputs
    output_round_mode: RoundMode | None = None

    def asdict(self) -> dict[str, Any]:
        """Return the data from this class as a dictionary.

        :return: a dictionary with each attribute and property of this dataclass as keys and the associated values
        """
        return asdict(self)
