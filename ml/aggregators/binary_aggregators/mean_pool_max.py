from rationai.mlkit.metrics.aggregators import MeanPoolMaxAggregator
from torch import Tensor

from ml.aggregators.binary_aggregators.base import BinaryAggregator
from ml.datamodule.datasets.base import Slide


class MeanPoolMaxBinaryAggregator(BinaryAggregator):
    def __init__(self, kernel_size: int, threshold: float = 0.5) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold

    def is_carcinoma(
        self,
        carcinoma_prob: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> bool:

        aggr = MeanPoolMaxAggregator(
            self.kernel_size, slide["tile_extent_x"], slide["stride_x"]
        )

        aggr.update(carcinoma_prob, carcinoma_prob, x=x, y=y)
        mean_pooled_max, _ = aggr.compute()
        return mean_pooled_max.item() > self.threshold
