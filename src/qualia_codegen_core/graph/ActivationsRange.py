from __future__ import annotations

from qualia_codegen_core.typing import TYPE_CHECKING

from .ActivationRange import ActivationRange
from .RoundMode import RoundMode

if TYPE_CHECKING:
    from pathlib import Path

class ActivationsRange(dict[str, ActivationRange]):
    def __int_or_none(self, s: str) -> int | None:
        if s == 'None':
            return None
        return int(s)

    def __roundmode_or_none(self, s: str) -> RoundMode | None:
        if s == 'None':
            return None
        return RoundMode(s)

    def load(self,
             path: Path,
             input_layer_name: str) -> ActivationsRange:
        first_input_bits: int | None = None
        first_input_q: int | None = None
        first_input_round_mode: RoundMode | None = None

        with path.open() as f:
            for line in f:
                r = line.strip().split(',')
                self[r[0]] = ActivationRange(self.__int_or_none(r[1]),  # input_bits
                                             self.__int_or_none(r[2]),  # activation_bits
                                             self.__int_or_none(r[3]),  # weights_bits
                                             self.__int_or_none(r[4]),  # bias_bits
                                             self.__int_or_none(r[5]),  # input_q
                                             self.__int_or_none(r[6]),  # activation_q
                                             self.__int_or_none(r[7]),  # weights_q
                                             self.__int_or_none(r[8]),  # bias_q
                                             self.__roundmode_or_none(r[9]),  # input_round_mode
                                             self.__roundmode_or_none(r[10]),  # activation_round_mode
                                             self.__roundmode_or_none(r[11]))  # weights_round_mode
                if first_input_bits is None:
                    first_input_bits = self.__int_or_none(r[1])
                if first_input_q is None:
                    first_input_q = self.__int_or_none(r[5])
                if first_input_round_mode is None:
                    first_input_round_mode = self[r[0]].input_round_mode

        # Model input range
        self[input_layer_name] = ActivationRange(first_input_bits,  # input_bits
                                                 first_input_bits,  # activation_bits
                                                 0,  # weights_bits
                                                 None,  # bias_bits
                                                 first_input_q,  # input_q
                                                 first_input_q,  # activation_q
                                                 0,  # weights_q
                                                 None,  # bias_q
                                                 first_input_round_mode,  # input_round_mode
                                                 first_input_round_mode,  # activation_round_mode
                                                 None)  # weights_round_mode
        return self
