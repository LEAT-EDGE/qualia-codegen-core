from dataclasses import dataclass

from qualia_codegen_core.typing import NDArrayFloatOrInt

from .TActivationLayer import TActivation
from .TBaseLayer import TBaseLayer


@dataclass
class TDenseLayer(TBaseLayer):
    activation: TActivation
    kernel: NDArrayFloatOrInt
    units: int
    use_bias: bool
    bias: NDArrayFloatOrInt

    @property
    def weights(self) -> dict[str, NDArrayFloatOrInt]:
        w = {'kernel': self.kernel}
        if self.use_bias:
            w['bias'] = self.bias
        return w
