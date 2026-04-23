from collections.abc import Sequence
from typing import cast

import numpy as np
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
            weights=self._get_weights(
                cast("list[EmbeddingsTileDataset]", slides_dataset.datasets)
            ),
            num_samples=num_samples,
            replacement=replacement,
        )

    @staticmethod
    def _get_weights(slides: list[EmbeddingsTileDataset]) -> Sequence[float]:

        tile_counts = np.array([len(s) for s in slides], dtype=np.int64)

        labels = []
        for s in slides:
            assert s.label is not None
            labels.append(s.label.item())

        _, label_indices = np.unique(labels, return_inverse=True)

        label_counts = np.bincount(label_indices, weights=tile_counts)
        label_weights = label_counts.sum() / label_counts
        slide_weights = label_weights[label_indices]

        return np.repeat(slide_weights, tile_counts).tolist()
