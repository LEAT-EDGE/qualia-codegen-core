from __future__ import annotations

from qualia_codegen_core.typing import TYPE_CHECKING

from .ActivationRange import ActivationRange
from .RoundMode import RoundMode

if TYPE_CHECKING:
    from pathlib import Path  # noqa: TCH003

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
        first_input_q: int | None = None
        first_input_round_mode: RoundMode | None = None

        with path.open() as f:
            for line in f:
                r = line.strip().split(',')
                self[r[0]] = ActivationRange(self.__int_or_none(r[1]),
                                             self.__int_or_none(r[2]),
                                             self.__int_or_none(r[3]),
                                             self.__int_or_none(r[4]),
                                             self.__roundmode_or_none(r[5]),
                                             self.__roundmode_or_none(r[6]),
                                             self.__roundmode_or_none(r[7]))
                if first_input_q is None:
                    first_input_q = self.__int_or_none(r[1])
                if first_input_round_mode is None:
                    first_input_round_mode = self[r[0]].input_round_mode

        # Model input range
        self[input_layer_name] = ActivationRange(first_input_q,
                                                 first_input_q,
                                                 0,
                                                 None,
                                                 first_input_round_mode,
                                                 first_input_round_mode,
                                                 None)
        return self
