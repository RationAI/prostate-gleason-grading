from rationai.mlkit.metrics.aggregators import TopKAggregator
from torch import Tensor

from ml.aggregators.binary_aggregators.base import BinaryAggregator
from ml.datamodule.datasets.base import Slide


class MeanPoolTopKBinaryAggregator(BinaryAggregator):
    def __init__(self, kernel_size: int, k: int, threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.k = k
        self.threshold = threshold

    def is_carcinoma(
        self,
        carcinoma_prob: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> bool:

        aggr = TopKAggregator(
            self.kernel_size, slide["tile_extent_x"], slide["stride_x"], self.k
        )

        aggr.update(carcinoma_prob, carcinoma_prob, x=x, y=y)
        mean_pooled_top_k, _ = aggr.compute()
        return mean_pooled_top_k.item() >= self.threshold
