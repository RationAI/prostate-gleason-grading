from abc import ABC, abstractmethod

from rationai.mlkit.metrics.aggregators import MeanPoolMaxAggregator, TopKAggregator
from torch import Tensor

from ml.datamodule.datasets.base import Slide


class BinaryAggregator(ABC):
    def __init__(self, prediction_threshold: float) -> None:
        self.prediction_threshold = prediction_threshold

    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> float | None:
        return self.aggregate(1 - probs[:, 0], x, y, slide)

    def is_carcinoma(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> bool | None:
        score = self(probs, x, y, slide)
        return None if score is None else score > self.prediction_threshold

    @abstractmethod
    def aggregate(
        self,
        carcinoma_prob: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> float | None:
        pass


class MaxBinaryAggregator(BinaryAggregator):
    def __init__(self, prediction_threshold: float) -> None:
        super().__init__(prediction_threshold)

    def aggregate(
        self,
        carcinoma_prob: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> float:
        return carcinoma_prob.max().item()


class MeanPoolMaxBinaryAggregator(BinaryAggregator):
    def __init__(self, prediction_threshold: float, kernel_size: int) -> None:
        super().__init__(prediction_threshold)
        self.kernel_size = kernel_size

    def aggregate(
        self,
        carcinoma_prob: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> float | None:
        aggr = MeanPoolMaxAggregator(
            self.kernel_size, slide["tile_extent_x"], slide["stride_x"]
        )

        aggr.update(carcinoma_prob, carcinoma_prob, x=x, y=y)
        try:
            return aggr.compute()[0].item()
        except RuntimeError as exc:
            if "Output size is too small" in str(exc):
                print(f"[WARN] {slide['stem']} to small; returning None.")
                return None
            else:
                raise


class MeanPoolTopKBinaryAggregator(BinaryAggregator):
    def __init__(self, prediction_threshold: float, kernel_size: int, k: int) -> None:
        super().__init__(prediction_threshold)
        self.kernel_size = kernel_size
        self.k = k

    def aggregate(
        self,
        carcinoma_prob: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> float | None:
        aggr = TopKAggregator(
            self.kernel_size, slide["tile_extent_x"], slide["stride_x"], self.k
        )

        aggr.update(carcinoma_prob, carcinoma_prob, x=x, y=y)
        try:
            return aggr.compute()[0].item()
        except RuntimeError as exc:
            if "Output size is too small" in str(exc):
                print(f"[WARN] {slide['stem']} to small; returning None.")
                return None
            else:
                raise
