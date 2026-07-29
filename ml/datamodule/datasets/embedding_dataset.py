from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, cast

import torch
from torch.utils.data import Dataset

from ml.datamodule.datasets.base import FilterableDataset, SlideTiles
from ml.typing import LabeledSample, Metadata, UnlabeledSample


T_co = TypeVar("T_co", covariant=True)


class EmbeddingsTileDataset(Dataset[LabeledSample | UnlabeledSample]):
    def __init__(self, slide_tiles: SlideTiles, embeddings_col: str) -> None:

        super().__init__()

        self.slide = slide_tiles.slide["stem"]
        self.label = slide_tiles.slide_label
        self.tiles = slide_tiles
        self.embeddings_col = embeddings_col

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> LabeledSample | UnlabeledSample:

        tile = self.tiles[idx]
        embedding = torch.as_tensor(tile[self.embeddings_col])
        metadata = Metadata(slide=self.slide, x=tile["x"], y=tile["y"])

        return (
            (embedding, metadata, self.tiles.tile_labels[idx])
            if self.tiles.tile_labels is not None
            else (embedding, metadata)
        )


class EmbeddingsSlideDataset(FilterableDataset[T_co]):
    def __init__(
        self,
        labeled: bool,
        embeddings_col: str,
        qc_and_tissue_thresholds: dict[str, float],
        carcinoma_prediction_threshold: float | None,
        apply_carcinoma_prediction_filter: bool,
        uris: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
        fold: int | None = None,
        invert_fold_selection: bool = False,
        labels_map: dict[str, int] | None = None,
    ) -> None:

        self.embeddings_col = embeddings_col

        super().__init__(
            labeled=labeled,
            qc_and_tissue_thresholds=qc_and_tissue_thresholds,
            carcinoma_prediction_threshold=carcinoma_prediction_threshold,
            apply_carcinoma_prediction_filter=apply_carcinoma_prediction_filter,
            uris=uris,
            paths=paths,
            fold=fold,
            invert_fold_selection=invert_fold_selection,
            labels_map=labels_map,
        )

    def _generate_slide_dataset(self, slide_tiles: SlideTiles) -> Dataset[T_co]:
        return cast(
            "Dataset[T_co]",
            EmbeddingsTileDataset(slide_tiles, self.embeddings_col),
        )


class LabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[LabeledSample]):
    def get_slide_labels(self) -> dict[str, torch.Tensor]:

        assert self.labeled, "SlideDataset is not labeled."

        labels: dict[str, torch.Tensor] = {}

        for dataset in self.datasets:
            slide = cast("EmbeddingsTileDataset", dataset)
            assert slide.label is not None
            labels[slide.slide] = slide.label

        return labels

    def get_tile_labels(self) -> torch.Tensor:

        assert self.labeled, "SlideDataset is not labeled."

        labels: list[torch.Tensor] = []

        for dataset in self.datasets:
            slide = cast("EmbeddingsTileDataset", dataset)
            assert slide.tiles.tile_labels is not None
            labels.append(slide.tiles.tile_labels)

        return torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)


class UnlabeledEmbeddingsSlideDataset(EmbeddingsSlideDataset[UnlabeledSample]): ...
