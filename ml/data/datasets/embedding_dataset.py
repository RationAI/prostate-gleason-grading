from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar, cast

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
        slide: dict[str, Any],
        tiles: HFDataset,
        embeddings_path: str,
        filtered_indices: np.ndarray,
        labeled: bool,
    ) -> None:

        super().__init__()

        self.slide = slide
        self.tiles = tiles
        self.filtered_indices = filtered_indices
        self.embeddings_path = embeddings_path
        self.labeled = labeled

        self.embeddings: torch.Tensor | None = None

    def _load_embeddings(self) -> None:
        if self.embeddings is None:
            embeddings = torch.load(self.embeddings_path, map_location="cpu")
            if len(embeddings) != len(self.tiles):
                raise ValueError(f"Slide {self.slide['stem']}: incompatible embeddings")
            self.embeddings = embeddings[self.filtered_indices]

    def __len__(self) -> int:
        return len(self.filtered_indices)

    def __getitem__(self, idx: int) -> LabeledSample | UnlabeledSample:

        if not 0 <= idx < len(self):
            raise ValueError(f"Slide {self.slide['stem']}: index out of range")

        self._load_embeddings()

        assert self.embeddings is not None

        tile = self.tiles[self.filtered_indices[idx]]
        embedding = self.embeddings[idx]
        metadata = Metadata(slide=self.slide["stem"], x=tile["x"], y=tile["y"])

        return (
            (embedding, metadata, self.slide["gleason_score"])
            if self.labeled
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
    ) -> None:

        if fold is not None and mode not in ["train", "val"]:
            raise ValueError(
                f"Invalid mode '{mode}': if fold is specified,"
                f"mode must be one of 'train' or 'val'"
            )

        self.fold = fold
        self.mode = mode
        self.embeddings_dir = embeddings_dir

        self.slides: HFDataset

        super().__init__(
            dataset_uris, qc_and_tissue_thresholds, carcinoma_prediction_threshold
        )

    def generate_datasets(self) -> Iterable[Dataset[T]]:

        if self.fold is not None:
            if self.fold not in self.slides.unique("fold"):
                raise ValueError(f"Unknown fold: {self.fold}")
            if self.mode == "train":
                self.slides = self.slides.filter(lambda s: s["fold"] != self.fold)
            elif self.mode == "val":
                self.slides = self.slides.filter(lambda s: s["fold"] == self.fold)

        for slide in self.slides:
            yield cast(
                "Dataset[T]",
                EmbeddingsTileDataset(
                    slide=slide,
                    tiles=self.filter_tiles_by_slide(slide["id"]),
                    filtered_indices=self.indices_of_filtered_tiles(slide),
                    labeled=self.labeled,
                    embeddings_path=(self.embeddings_dir / slide["stem"]).with_suffix(
                        ".pt"
                    ),
                ),
            )


class LabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[LabeledSample]): ...


class UnlabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[UnlabeledSample]): ...
