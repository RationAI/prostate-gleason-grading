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
        embedding = torch.as_tensor(tile[self.embeddings_col])
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
                label = torch.tensor(
                    self.labels_map[slide["gleason_score"]],
                    dtype=torch.long,
                )

            tiles = self.filter_tiles_by_slide_and_thresholds(slide)

            if len(tiles) == 0:
                print(
                    f"Warning: slide {slide['stem']} has no tiles "
                    f"left after filtering - it will be skipped"
                )
                continue

            yield cast(
                "Dataset[T]",
                EmbeddingsTileDataset(
                    slide=slide["stem"],
                    tiles=tiles,
                    label=label,
                    embeddings_col=self.embeddings_col,
                ),
            )


class LabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[LabeledSample]):
    def get_labels(self) -> torch.Tensor:

        assert self.labeled, "SlideDataset is not labeled."

        slide_labels: list[int] = []
        slide_lengths: list[int] = []

        for dataset in self.datasets:
            slide = cast("EmbeddingsTileDataset", dataset)
            assert slide.label is not None, f"Slide {slide.slide}: unknown label."
            slide_labels.append(int(slide.label.item()))
            slide_lengths.append(len(slide))

        return torch.repeat_interleave(
            torch.tensor(slide_labels, dtype=torch.long),
            torch.tensor(slide_lengths, dtype=torch.long),
        )


class UnlabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[UnlabeledSample]): ...
