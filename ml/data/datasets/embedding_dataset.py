from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from ml.data.datasets.base import FilterableDataset
from ml.typing import LabeledSample, Metadata, UnlabeledSample


T = TypeVar("T", covariant=True)


class EmbeddingsTileDataset(Dataset[LabeledSample | UnlabeledSample]):
    def __init__(
        self,
        slide: str,
        tiles: HFDataset,
        embeddings_path: Path,
        filtered_indices: np.ndarray,
        label: torch.Tensor | None = None,
    ) -> None:

        super().__init__()

        self.slide = slide
        self.tiles = tiles
        self.filtered_indices = filtered_indices
        self.embeddings_path = str(embeddings_path)
        self.label = label

        self.embeddings: torch.Tensor | None = None

    def _load_embeddings(self) -> None:
        if self.embeddings is None:
            embeddings = torch.load(self.embeddings_path, map_location="cpu")
            if len(embeddings) != len(self.tiles):
                raise ValueError(f"Slide {self.slide}: incompatible embeddings")
            self.embeddings = embeddings[self.filtered_indices]

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, idx: int) -> LabeledSample | UnlabeledSample:

        if not 0 <= idx < len(self):
            raise IndexError(f"Slide {self.slide}: index out of range")

        self._load_embeddings()

        assert self.embeddings is not None

        tile = self.tiles[self.filtered_indices[idx]]
        embedding = self.embeddings[idx]
        metadata = Metadata(slide=self.slide, x=tile["x"], y=tile["y"])

        return (
            (embedding, metadata, self.label)
            if self.label is not None
            else (embedding, metadata)
        )


class EmbeddingsSlideDataset(FilterableDataset[T]):
    def __init__(
        self,
        dataset_uris: Iterable[str],
        embeddings_dir: Path,
        qc_and_tissue_thresholds: dict[str, float],
        carcinoma_prediction_threshold: float | None = None,
        fold: int | None = None,
        mode: str | None = None,
        labels_map: dict[str, int] | None = None,
    ) -> None:

        self.embeddings_dir = embeddings_dir

        self.slides: HFDataset

        super().__init__(
            dataset_uris,
            qc_and_tissue_thresholds,
            carcinoma_prediction_threshold,
            fold,
            mode,
            labels_map,
        )

    def generate_datasets(self) -> Iterable[Dataset[T]]:

        self.filter_slides_by_fold()

        for slide in self.slides:
            label = None

            if self.labeled:
                assert self.labels_map is not None
                label = torch.tensor(self.labels_map[slide["gleason_score"]])

            yield cast(
                "Dataset[T]",
                EmbeddingsTileDataset(
                    slide=slide["stem"],
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                    filtered_indices=self.indices_of_filtered_tiles(slide),
                    label=label,
                    embeddings_path=(self.embeddings_dir / slide["stem"]).with_suffix(
                        ".pt"
                    ),
                ),
            )


class LabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[LabeledSample]): ...


class UnlabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[UnlabeledSample]): ...
