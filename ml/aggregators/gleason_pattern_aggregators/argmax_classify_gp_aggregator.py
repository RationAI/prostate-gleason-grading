import torch
from rationai.masks import HeatmapAssembler
from torch import Tensor, nn

from ml.aggregators.gleason_pattern_aggregators.base import (
    DominantGleasonPatternAggregator,
)
from ml.data_module.datasets.base import Slide


class ArgmaxClassifyGleasonPatternAggregator(DominantGleasonPatternAggregator):
    def __init__(
        self,
        kernel_size: int = 1,
        min_percentage_for_secondary: float = 0.05,
    ) -> None:
        self.kernel_size = kernel_size
        self.min_percentage_for_secondary = min_percentage_for_secondary

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

    def __call__(
        self,
        probs: Tensor,
        x: Tensor,
        y: Tensor,
        slide: Slide,
    ) -> tuple[int, int]:

        num_classes = probs.shape[1]

        class_heatmaps: list[Tensor] = []
        for class_idx in range(num_classes):
            class_heatmaps.append(self._get_heatmap(probs[:, class_idx], x, y, slide))

        class_map = torch.stack(class_heatmaps, dim=0).argmax(dim=0)

        gp_counts = torch.bincount(class_map.reshape(-1), minlength=num_classes)
        gp_counts = gp_counts[1:]  # drop negative tiles
        if gp_counts.sum() == 0:
            raise RuntimeError(f"Slide {slide['stem']} contains no GP3+.")
        gp_percentages = gp_counts.float() / gp_counts.sum()

        dominant_gp = int(gp_percentages.argmax().item())

        candidates = torch.where(gp_percentages >= self.min_percentage_for_secondary)[0]
        candidates = candidates[candidates != dominant_gp]

        if len(candidates) == 0:
            secondary_gp = dominant_gp
        else:
            secondary_gp = int(candidates[gp_counts[candidates].argmax()].item())

        return dominant_gp + 3, secondary_gp + 3  # the minimal assined GP is 3
