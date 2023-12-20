import sys
from abc import ABC
from collections import OrderedDict
from dataclasses import dataclass

from qualia_codegen_core.typing import DTypes, NDArrayFloatOrInt, Shapes

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass(eq=False)
class TBaseLayer(ABC):
    input_shape: Shapes
    output_shape: Shapes
    output_dtype: DTypes
    name: str

    @override
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @property
    def weights(self) -> OrderedDict[str, NDArrayFloatOrInt]:
        # If adding any weights in a child layer, fill dict with same name as attribute
        return OrderedDict()
