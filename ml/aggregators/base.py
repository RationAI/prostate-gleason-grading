from abc import ABC, abstractmethod

from torch import Tensor


class Aggregator(ABC):
    @abstractmethod
    def num_output_classes(self, num_tile_classes: int) -> int:
        pass

    @abstractmethod
    def __call__(self, preds: Tensor, x: Tensor, y: Tensor) -> tuple[Tensor, str]:
        pass
