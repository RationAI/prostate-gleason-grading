from abc import ABC, abstractmethod

from torch import Tensor

from ml.aggregators.binary_aggregators import BinaryAggregator
from ml.aggregators.gleason_pattern_aggregators import DominantGleasonPatternAggregator
from ml.data_module.datasets.base import Slide


class GleasonScoreAggregator(ABC):
    def __init__(
        self,
        binary_aggregator: BinaryAggregator,
        pattern_aggregator: DominantGleasonPatternAggregator,
    ) -> None:
        self.binary_aggregator = binary_aggregator
        self.pattern_aggregator = pattern_aggregator

    @abstractmethod
    def num_output_classes(self, num_tile_classes: int) -> int:
        pass

    @abstractmethod
    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> tuple[Tensor, str]:
        pass


class PureGleasonScoreAggregator(GleasonScoreAggregator):
    def num_output_classes(self, num_tile_classes: int) -> int:
        return num_tile_classes

    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> tuple[Tensor, str]:

        if not self.binary_aggregator(probs, x, y, slide):
            return Tensor(0), "None"

        primary_pattern, _ = self.pattern_aggregator(probs, x, y, slide)
        gleason_score = f"{primary_pattern}+{primary_pattern}"
        return Tensor(primary_pattern - 2), gleason_score


class MixedGleasonScoreAggregator(GleasonScoreAggregator):
    def __init__(
        self,
        labels_map: dict[str, int],
        binary_aggregator: BinaryAggregator,
        pattern_aggregator: DominantGleasonPatternAggregator,
    ) -> None:
        super().__init__(binary_aggregator, pattern_aggregator)
        self.labels_map = labels_map

    def num_output_classes(self, num_tile_classes: int) -> int:
        return len(set(self.labels_map.values()))

    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> tuple[Tensor, str]:

        if not self.binary_aggregator(probs, x, y, slide):
            return Tensor(0), "None"

        primary_pattern, secondary_pattern = self.pattern_aggregator(probs, x, y, slide)
        gleason_score = f"{primary_pattern}+{secondary_pattern}"
        return Tensor(self.labels_map[gleason_score]), gleason_score
