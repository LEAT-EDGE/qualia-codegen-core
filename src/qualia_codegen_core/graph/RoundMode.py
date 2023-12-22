from __future__ import annotations

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if sys.version_info >= (3, 11):
    from enum import StrEnum, auto
else:
    from enum import Enum, auto
    class StrEnum(str, Enum):
        @override
        @staticmethod
        def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
            return name.lower()


class RoundMode(StrEnum):
    NONE = auto()
    FLOOR = auto()
    NEAREST = auto()

    @override
    @classmethod
    def _missing_(cls, value: object) -> RoundMode | None:
        if not isinstance(value, str):
            return None
        for member in cls:
            if member.value.upper() == value:
                return member
        return None
