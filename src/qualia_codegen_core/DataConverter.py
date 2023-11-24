# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from qualia_codegen_core.typing import NDArrayFloatOrInt


class DataConverter:
    dtype2ctype: Final[dict[DTypeLike, str]] = {
            np.float16: 'float',
            np.float32: 'float',
            np.float64: 'float',
            np.int8: 'int8_t',
            np.int16: 'int16_t',
            np.int32: 'int32_t',
            np.int64: 'int64_t',
            np.uint8: 'uint8_t',
            np.uint16: 'uint16_t',
            np.uint32: 'uint32_t',
            np.uint64: 'uint64_t',
            }

    def qtype2ctype(self, number_type: type[int |float], width: int) -> str:
        if number_type == int:
            return f'{number_type.__name__}{width}_t'
        if number_type == float:
            return number_type.__name__
        raise NotImplementedError

    def ndarray2cinitializer(self, arr: NDArrayFloatOrInt) -> str:
        if np.ndim(arr) == 0:
            # If float, use hex for exact representation, gets promoted to float64 from float32 but no big deal
            if np.issubdtype(arr.dtype, np.floating):
                return float(arr).hex()
            return str(arr)

        return '{' + ', '.join([self.ndarray2cinitializer(subarr) for subarr in arr]) + '}\n'

    def tensor2carray(self, arr: NDArrayFloatOrInt, name: str) -> dict[str, str | tuple[int, ...]]:
        arrdata = self.ndarray2cinitializer(arr)

        return {'name': name, 'data': arrdata, 'dtype': self.dtype2ctype[arr.dtype.type], 'shape': arr.shape}
