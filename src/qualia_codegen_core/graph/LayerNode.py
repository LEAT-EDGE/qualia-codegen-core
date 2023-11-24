from dataclasses import dataclass, field

from qualia_codegen_core.typing import Shapes

from .layers.TBaseLayer import TBaseLayer
from .Quantization import Quantization


@dataclass(eq=False)
class LayerNode:
    layer: TBaseLayer
    innodes: list['LayerNode'] = field(default_factory=list)
    outnodes: list['LayerNode'] = field(default_factory=list)
    q: Quantization = field(default_factory=Quantization)

    @property
    def input_shape(self) -> Shapes:
        return self.layer.input_shape

    @property
    def output_shape(self) -> Shapes:
        return self.layer.output_shape

    def __repr__(self) -> str:
        return self.layer.name
