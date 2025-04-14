import os
import typing
from typing import Any, Optional, TypeVar, Union

import numpy as np
from numpy import typing as npt

TYPE_CHECKING = typing.TYPE_CHECKING or os.environ.get('SPHINX_AUTODOC', None)
TBits = TypeVar('TBits', bound=npt.NBitBase)
NDArrayFloatOrInt = Union[npt.NDArray[np.floating[Any]], npt.NDArray[np.integer[Any]]]

class Shape(tuple[int, ...]):
    __slots__ = ()

class ShapeOptional(tuple[Optional[int], ...]):
    __slots__ = ()

class Shapes(tuple[Shape, ...]):
    __slots__ = ()

class DTypes(tuple[npt.DTypeLike, ...]):
    __slots__ = ()

__all__ = [
    'TYPE_CHECKING',
    'DTypes',
    'NDArrayFloatOrInt',
    'Shape',
    'ShapeOptional',
    'Shapes',
    'TBits',
]
