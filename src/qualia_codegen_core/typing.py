import os
import typing
from typing import Any, Optional, TypeVar, Union

import numpy as np
import numpy.typing

TYPE_CHECKING = typing.TYPE_CHECKING or os.environ.get('SPHINX_AUTODOC', False)
TBits = TypeVar('TBits', bound=numpy.typing.NBitBase)
NDArrayFloatOrInt = Union[numpy.typing.NDArray[np.floating[Any]], numpy.typing.NDArray[np.integer[Any]]]

class Shape(tuple[int, ...]):
    __slots__ = ()

class ShapeOptional(tuple[Optional[int], ...]):
    __slots__ = ()

class Shapes(tuple[Shape, ...]):
    __slots__ = ()

class DTypes(tuple[numpy.typing.DTypeLike, ...]):
    __slots__ = ()

__all__ = [
        'TYPE_CHECKING',
        'TBits',
        'NDArrayFloatOrInt',
        'Shape',
        'ShapeOptional',
        'Shapes',
        'DTypes',
        ]
