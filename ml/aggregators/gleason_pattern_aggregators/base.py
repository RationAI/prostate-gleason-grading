from abc import ABC, abstractmethod

from torch import Tensor

from ml.data_module.datasets.base import Slide


class DominantGleasonPatternAggregator(ABC):
    @abstractmethod
    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> tuple[int, int]:
        pass
