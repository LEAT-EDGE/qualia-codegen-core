from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

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
class TBatchNormalizationLayer(TBaseLayer):
    activation: TActivation
    mean: NDArrayFloatOrInt
    variance: NDArrayFloatOrInt
    gamma: NDArrayFloatOrInt
    beta: NDArrayFloatOrInt
    epsilon: NDArrayFloatOrInt

    _kernel: NDArrayFloatOrInt | None = None
    _bias: NDArrayFloatOrInt | None = None

    @property
    def kernel(self) -> NDArrayFloatOrInt:
        if self._kernel is None:
            stdev = np.sqrt(self.variance + self.epsilon)
            self._kernel = self.gamma / stdev
        return self._kernel

    @kernel.setter
    def kernel(self, v: NDArrayFloatOrInt) -> None:
        self._kernel = v

    @property
    def bias(self) -> NDArrayFloatOrInt:
        if self._bias is None:
            stdev = np.sqrt(self.variance + self.epsilon)
            self._bias = self.beta - self.gamma * self.mean / stdev
        return self._bias

    @bias.setter
    def bias(self, v: NDArrayFloatOrInt) -> None:
        self._bias = v

    @property
    @override
    def weights(self) -> OrderedDict[str, NDArrayFloatOrInt]:
        w = super().weights
        w['kernel'] = self.kernel
        w['bias'] = self.bias
        return w
