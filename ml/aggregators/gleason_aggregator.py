from abc import abstractmethod

import torch
from rationai.masks import HeatmapAssembler
from torch import Tensor, nn

from ml.aggregators.binary_aggregator import BinaryAggregator
from ml.datamodule.datasets.base import Slide


class GleasonScoreAggregator:
    def __init__(
        self,
        labels_map: dict[str, int],
        binary_aggregator: BinaryAggregator,
        noise_eps: float = 0.005,
        clinical_eps: float = 0.05,
        kernel_size: int = 1,
    ) -> None:
        self.labels_map = labels_map
        self.binary_aggregator = binary_aggregator
        self.noise_eps = noise_eps
        self.clinical_eps = clinical_eps
        self.kernel_size = kernel_size

    def num_classes(self) -> int:
        return len(set(self.labels_map.values()))

    def _get_heatmap(
        self,
        preds: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> Tensor:

        heatmap = HeatmapAssembler(
            extent_x=slide["extent_x"],
            extent_y=slide["extent_y"],
            extent_tile_x=slide["tile_extent_x"],
            extent_tile_y=slide["tile_extent_y"],
            stride_x=slide["stride_x"],
            stride_y=slide["stride_y"],
            device=str(preds.device),
        )

        heatmap.update(preds, x, y)
        computed = heatmap.compute()

        if self.kernel_size == 1:
            return computed

        pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=1)
        return pool(computed.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)

    @abstractmethod
    def pattern_percentages(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> Tensor:
        pass

    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> tuple[Tensor, str]:

        if not self.binary_aggregator.is_carcinoma(probs, x, y, slide):
            return torch.tensor([0], dtype=torch.long), "None"

        gp_percentages = self.pattern_percentages(probs, x, y, slide)

        # the clinical workflow for core needle biopsies:
        # - primary pattern = predominant by area
        # - secondary pattern = highest grade lesser pattern
        # - ignore lower grade pattern constituting less than
        #   5% of the overall tumor in high grade cases

        present_patterns = torch.where(gp_percentages > self.noise_eps)[0]
        if len(present_patterns) == 0:
            print(f"Slide {slide['stem']} marked as carcinoma, but contains no GP.")
            return torch.tensor([0], dtype=torch.long), "None"

        predominant_pattern = int(gp_percentages.argmax().item())

        lesser_patterns = present_patterns[present_patterns != predominant_pattern]

        ignore_mask = (
            (lesser_patterns < predominant_pattern)  # low grade in high grade case
            & (
                gp_percentages[lesser_patterns] < self.clinical_eps
            )  # and constitutes less than 5%
        )

        lesser_patterns = lesser_patterns[~ignore_mask]

        primary_pattern = 3 + predominant_pattern  # GP3 is the lowest

        if len(lesser_patterns) == 0:
            secondary_pattern = primary_pattern
        else:
            secondary_pattern = 3 + int(lesser_patterns[-1].item())

        gleason_score = f"{primary_pattern}+{secondary_pattern}"
        return torch.tensor(
            [self.labels_map[gleason_score]], dtype=torch.long
        ), gleason_score


class PatternSumGleasonScoreAggregator(GleasonScoreAggregator):
    def pattern_percentages(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> Tensor:

        gp_sums = probs[:, 1:].sum(dim=0).float()

        if gp_sums.sum() == 0:
            return torch.zeros_like(gp_sums, dtype=torch.float32)

        return gp_sums.float() / gp_sums.sum()


class ArgmaxClassifyGleasonScoreAggregator(GleasonScoreAggregator):
    def pattern_percentages(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> Tensor:

        num_classes = probs.shape[1]

        class_heatmaps: list[Tensor] = []
        for class_idx in range(num_classes):
            class_heatmaps.append(self._get_heatmap(probs[:, class_idx], x, y, slide))

        class_map = torch.stack(class_heatmaps, dim=0).argmax(dim=0)

        gp_counts = torch.bincount(class_map.reshape(-1), minlength=num_classes)
        gp_counts = gp_counts[1:]  # drop negative tiles

        if gp_counts.sum() == 0:
            return torch.zeros_like(gp_counts, dtype=torch.float32)

        return gp_counts.float() / gp_counts.sum()


class TumorWeightedGleasonScoreAggregator(GleasonScoreAggregator):
    def pattern_percentages(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> Tensor:

        num_classes = probs.shape[1]

        tumor_heatmap = 1.0 - self._get_heatmap(probs[:, 0], x, y, slide)

        pattern_sums: list[Tensor] = []
        for class_idx in range(1, num_classes):
            pattern_heatmap = self._get_heatmap(probs[:, class_idx], x, y, slide)
            pattern_sums.append((pattern_heatmap * tumor_heatmap).sum())

        gp_sums = torch.stack(pattern_sums)

        if gp_sums.sum() == 0:
            return torch.zeros_like(gp_sums, dtype=torch.float32)

        return gp_sums / gp_sums.sum()
