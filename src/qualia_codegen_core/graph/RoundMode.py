"""Contains enumeration of available rounding modes."""

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
        """Implements the StrEnum type missing from Python <= 3.10."""

        @override
        @staticmethod
        def _generate_next_value_(name: str, start: int, count: int, last_values: list[str]) -> str:
            return name.lower()


class RoundMode(StrEnum):
    """Enumerates available rounding modes."""

    #: No rounding, valid only for float data type
    NONE = auto()
    #: Rounding mode to round towards negative infinity
    FLOOR = auto()
    #: Rounding mode to round to nearest
    NEAREST = auto()

    @override
    @classmethod
    def _missing_(cls, value: object) -> RoundMode | None:
        """Convert a string containing the name of a rounding mode to its corresponding enumeration value.

        :param value: string containing the name of a rounding mode as a member of this enumeration
        :return: enumeration value for the given name, ``None`` if name is invalid
        """
        if not isinstance(value, str):
            return None
        for member in cls:
            if member.value.upper() == value:
                return member
        return None
