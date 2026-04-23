from typing import Any

import torch
from torch.utils.data import WeightedRandomSampler

from ml.datamodule.datasets.embedding_dataset import (
    EmbeddingsTileDataset,
    LabeledEmbeddingsSlideDataset,
)


class StratifiedWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        slides_dataset: LabeledEmbeddingsSlideDataset,
        num_samples: int | None = None,
        replacement: bool = True,
    ) -> None:

        if num_samples is None:
            num_samples = len(slides_dataset)
        elif not replacement and num_samples > len(slides_dataset):
            raise ValueError(
                "The number of samples can't exceed the size of the "
                "dataset when samples are drawn without replacement."
            )

        super().__init__(
            weights=self._get_weights(slides_dataset.datasets),
            num_samples=num_samples,
            replacement=replacement,
        )

    @staticmethod
    def _get_weights(slides: list[EmbeddingsTileDataset]) -> torch.Tensor:

        label_counts: dict[Any, int] = {}

        for slide in slides:
            label_counts[slide.label] = label_counts.get(slide.label, 0) + len(slide)

        total_count = sum(label_counts.values())

        label_probs_inverse = {
            label: total_count / label_count
            for label, label_count in label_counts.items()
        }

        return torch.repeat_interleave(
            torch.tensor(
                [label_probs_inverse[s.label] for s in slides],
                dtype=torch.double,
            ),
            torch.tensor([len(s) for s in slides], dtype=torch.int64),
        )
