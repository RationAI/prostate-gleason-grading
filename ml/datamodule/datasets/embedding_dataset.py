from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

from ml.datamodule.datasets.base import FilterableDataset
from ml.typing import LabeledSample, Metadata, UnlabeledSample


T = TypeVar("T", covariant=True)


class EmbeddingsTileDataset(Dataset[LabeledSample | UnlabeledSample]):
    def __init__(
        self,
        slide: str,
        tiles: HFDataset,
        embeddings_col: str,
        label: torch.Tensor | None = None,
    ) -> None:

        super().__init__()

        self.slide = slide
        self.tiles = tiles
        self.label = label
        self.embeddings_col = embeddings_col

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> LabeledSample | UnlabeledSample:

        if not 0 <= idx < len(self):
            raise IndexError(f"Slide {self.slide}: index out of range")

        tile = self.tiles[idx]
        embedding = torch.tensor(tile[self.embeddings_col])
        metadata = Metadata(slide=self.slide, x=tile["x"], y=tile["y"])

        return (
            (embedding, metadata, self.label)
            if self.label is not None
            else (embedding, metadata)
        )


class EmbeddingsSlideDataset(FilterableDataset[T]):
    def __init__(
        self,
        embeddings_col: str,
        qc_and_tissue_thresholds: dict[str, float],
        carcinoma_prediction_threshold: float | None = None,
        uris: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
        fold: int | None = None,
        mode: str | None = None,
        labels_map: dict[str, int] | None = None,
    ) -> None:

        self.embeddings_col = embeddings_col

        super().__init__(
            qc_and_tissue_thresholds=qc_and_tissue_thresholds,
            carcinoma_prediction_threshold=carcinoma_prediction_threshold,
            uris=uris,
            paths=paths,
            fold=fold,
            mode=mode,
            labels_map=labels_map,
        )

    def generate_datasets(self) -> Iterable[Dataset[T]]:

        if self.labeled:
            self._check_labels()

        if self.embeddings_col not in self.tiles.column_names:
            raise ValueError(f"Embeddings column '{self.embeddings_col}' is missing")

        for slide in self.filter_slides_by_fold():
            label = None

            if self.labeled:
                assert self.labels_map is not None
                label = torch.tensor(self.labels_map[slide["gleason_score"]])

            yield cast(
                "Dataset[T]",
                EmbeddingsTileDataset(
                    slide=slide["stem"],
                    tiles=self.filter_tiles_by_slide_and_thresholds(slide),
                    label=label,
                    embeddings_col=self.embeddings_col,
                ),
            )


class LabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[LabeledSample]): ...


class UnlabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[UnlabeledSample]): ...
