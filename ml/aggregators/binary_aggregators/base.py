from abc import ABC, abstractmethod

from torch import Tensor

from ml.data_module.datasets.base import Slide


class BinaryAggregator(ABC):
    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> bool:
        return self.is_carcinoma(1 - probs[:, 0], x, y, slide)

    @abstractmethod
    def is_carcinoma(
        self,
        carcinoma_prob: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> bool:
        pass
