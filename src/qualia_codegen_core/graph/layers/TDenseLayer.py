from __future__ import annotations

import sys
from dataclasses import dataclass

from qualia_codegen_core.typing import TYPE_CHECKING, NDArrayFloatOrInt

from .TBaseLayer import TBaseLayer

if TYPE_CHECKING:
    from collections import OrderedDict  # noqa: TCH003

    from .TActivationLayer import TActivation  # noqa: TCH001

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

@dataclass
class TDenseLayer(TBaseLayer):
    activation: TActivation
    kernel: NDArrayFloatOrInt
    units: int
    use_bias: bool
    bias: NDArrayFloatOrInt

    @property
    @override
    def weights(self) -> OrderedDict[str, NDArrayFloatOrInt]:
        w = super().weights
        w['kernel'] = self.kernel
        if self.use_bias:
            w['bias'] = self.bias
        return w
