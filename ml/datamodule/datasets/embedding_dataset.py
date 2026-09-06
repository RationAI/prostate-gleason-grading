from typing import Any

import torch

from ml.datamodule.datasets.base import (
    LabeledSlideDataset,
    LabeledTileDataset,
    Slide,
    Tiles,
    UnlabeledSlideDataset,
    UnlabeledTileDataset,
)
from ml.typing import LabeledSample, Metadata, UnlabeledSample


class EmbeddingsTileDatasetMixin:
    slide: Slide
    tiles: Tiles
    embeddings_col: str

    def _get_embedding_and_metadata(self, idx: int) -> tuple[torch.Tensor, Metadata]:
        tile = self.tiles[idx]
        return (
            torch.as_tensor(tile[self.embeddings_col]),
            Metadata(slide=self.slide["stem"], x=tile["x"], y=tile["y"]),
        )


class UnlabeledEmbeddingsTileDataset(
    UnlabeledTileDataset[UnlabeledSample], EmbeddingsTileDatasetMixin
):
    def __init__(
        self,
        slide: Slide,
        tiles: Tiles,
        embeddings_col: str,
    ) -> None:
        super().__init__(slide, tiles)
        self.embeddings_col = embeddings_col

    def __getitem__(self, idx: int) -> UnlabeledSample:
        return self._get_embedding_and_metadata(idx)


class LabeledEmbeddingsTileDataset(
    LabeledTileDataset[LabeledSample], EmbeddingsTileDatasetMixin
):
    def __init__(
        self,
        slide: Slide,
        tiles: Tiles,
        slide_label: torch.Tensor,
        tile_labels: torch.Tensor,
        embeddings_col: str,
    ) -> None:
        super().__init__(slide, tiles, slide_label, tile_labels)
        self.embeddings_col = embeddings_col

    def __getitem__(self, idx: int) -> LabeledSample:
        embedding, metadata = self._get_embedding_and_metadata(idx)
        return embedding, metadata, self.tile_labels[idx]


class UnlabeledEmbeddingsSlideDataset(UnlabeledSlideDataset[UnlabeledSample]):
    def __init__(self, embeddings_col: str, **kwargs: Any) -> None:
        self.embeddings_col = embeddings_col
        super().__init__(**kwargs)

    def _generate_tile_dataset(
        self, slide: Slide, tiles: Tiles
    ) -> UnlabeledEmbeddingsTileDataset:
        return UnlabeledEmbeddingsTileDataset(slide, tiles, self.embeddings_col)


class LabeledEmbeddingsSlideDataset(LabeledSlideDataset[LabeledSample]):
    def __init__(self, embeddings_col: str, **kwargs: Any) -> None:
        self.embeddings_col = embeddings_col
        super().__init__(**kwargs)

    def _generate_tile_dataset(
        self,
        slide: Slide,
        tiles: Tiles,
        slide_label: torch.Tensor,
        tile_labels: torch.Tensor,
    ) -> LabeledEmbeddingsTileDataset:
        return LabeledEmbeddingsTileDataset(
            slide,
            tiles,
            slide_label,
            tile_labels,
            self.embeddings_col,
        )
