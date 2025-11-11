from __future__ import annotations
from typing import Protocol, Tuple
import numpy as np

class DGPProtocol(Protocol):
    def __call__(self, n: int, p: int, s: int, beta1: float, rho: float, seed: int | None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...

class EstimatorProtocol(Protocol):
    def __call__(self, Y: np.ndarray, D: np.ndarray, X: np.ndarray, **kwargs) -> dict: ...
