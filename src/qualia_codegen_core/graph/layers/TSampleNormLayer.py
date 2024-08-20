from dataclasses import dataclass
from enum import Enum

from .TBaseLayer import TBaseLayer


class TSampleNormMode(Enum):
    MINMAX = 0
    ZSCORE = 1

@dataclass
class TSampleNormLayer(TBaseLayer):
    mode: TSampleNormMode
