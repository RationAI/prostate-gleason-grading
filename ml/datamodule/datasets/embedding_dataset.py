from typing import Any

import torch
from torch import Tensor

from ml.datamodule.datasets.base import (
    FullyLabeledSlideDataset,
    FullyLabeledTileDataset,
    Slide,
    Tiles,
    UnlabeledSlideDataset,
    UnlabeledTileDataset,
    WeaklyLabeledSlideDataset,
    WeaklyLabeledTileDataset,
)
from ml.typing import BagMetadata, Metadata


class EmbeddingsTileDataset:
    slide: Slide
    tiles: Tiles

    def __init__(self, embeddings_col: str, **kwargs: Any) -> None:
        self.embeddings_col = embeddings_col
        super().__init__(**kwargs)

    def _get_sample(self, idx: int) -> tuple[Tensor, Metadata]:
        tile = self.tiles[idx]
        return (
            torch.as_tensor(tile[self.embeddings_col]),
            Metadata(slide=self.slide["stem"], x=tile["x"], y=tile["y"]),
        )

    def _get_bag(self) -> tuple[Tensor, BagMetadata]:
        tiles = self.tiles.columns_as_tensors([self.embeddings_col, "x", "y"])
        return (
            tiles[self.embeddings_col],
            BagMetadata(slide=self.slide["stem"], x=tiles["x"], y=tiles["y"]),
        )


class EmbeddingsSlideDataset:
    def __init__(self, embeddings_col: str, **kwargs: Any) -> None:
        self.embeddings_col = embeddings_col
        super().__init__(**kwargs)


class UnlabeledEmbeddingsTileDataset(EmbeddingsTileDataset, UnlabeledTileDataset): ...


class WeaklyLabeledEmbeddingsTileDataset(
    EmbeddingsTileDataset,
    WeaklyLabeledTileDataset,
): ...


class FullyLabeledEmbeddingsTileDataset(
    EmbeddingsTileDataset,
    FullyLabeledTileDataset,
): ...


class UnlabeledEmbeddingsSlideDataset(
    EmbeddingsSlideDataset,
    UnlabeledSlideDataset,
):
    def _generate_tile_dataset(
        self, slide: Slide, tiles: Tiles
    ) -> UnlabeledEmbeddingsTileDataset:
        return UnlabeledEmbeddingsTileDataset(
            self.embeddings_col,
            slide=slide,
            tiles=tiles,
        )


class WeaklyLabeledEmbeddingsSlideDataset(
    EmbeddingsSlideDataset,
    WeaklyLabeledSlideDataset,
):
    def _generate_tile_dataset(
        self,
        slide: Slide,
        tiles: Tiles,
        slide_label: Tensor,
    ) -> WeaklyLabeledEmbeddingsTileDataset:
        return WeaklyLabeledEmbeddingsTileDataset(
            self.embeddings_col,
            slide=slide,
            tiles=tiles,
            slide_label=slide_label,
        )


class FullyLabeledEmbeddingsSlideDataset(
    EmbeddingsSlideDataset,
    FullyLabeledSlideDataset,
):
    def _generate_tile_dataset(
        self,
        slide: Slide,
        tiles: Tiles,
        slide_label: Tensor,
        tile_labels: Tensor,
    ) -> FullyLabeledEmbeddingsTileDataset:
        return FullyLabeledEmbeddingsTileDataset(
            self.embeddings_col,
            slide=slide,
            tiles=tiles,
            slide_label=slide_label,
            tile_labels=tile_labels,
        )
