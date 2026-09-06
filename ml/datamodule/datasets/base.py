from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar, override

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset


type Slide = dict[str, Any]
type Tile = dict[str, Any]

T_co = TypeVar("T_co", covariant=True)


class Tiles:
    def __init__(self, tiles: HFDataset, tile_indices: np.ndarray) -> None:
        self._tiles = tiles
        self._index_map = tile_indices

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> Tile:
        return self._tiles[self._index_map[idx]]


class TileDataset(Dataset[T_co], ABC):
    def __init__(self, slide: Slide, tiles: Tiles) -> None:
        super().__init__()
        self.slide = slide
        self.tiles = tiles

    def __len__(self) -> int:
        return len(self.tiles)

    @abstractmethod
    def __getitem__(self, idx: int) -> T_co:
        pass


class UnlabeledTileDataset(TileDataset[T_co], ABC): ...


class LabeledTileDataset(TileDataset[T_co], ABC):
    def __init__(
        self,
        slide: Slide,
        tiles: Tiles,
        slide_label: torch.Tensor,
        tile_labels: torch.Tensor,
    ) -> None:

        if slide_label.ndim != 0:
            raise ValueError("Scalar tensor is expected as a slide label.")

        if len(tiles) != tile_labels.numel():
            raise ValueError(
                "The number of tile labels must match the number of tiles."
            )

        super().__init__(slide, tiles)
        self.slide_label = slide_label
        self.tile_labels = tile_labels


class SlideDataset(MetaTiledSlides[T_co], ABC):
    slides: HFDataset
    tiles: HFDataset

    def __init__(
        self,
        qc_and_tissue_thresholds: dict[str, float],
        fold: int | None = None,
        invert_fold_selection: bool = False,
        uris: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
    ) -> None:

        self.fold = fold
        self.invert_fold_selection = invert_fold_selection

        self.qc_and_tissue_thresholds = qc_and_tissue_thresholds
        self._qc_and_tissue_mask: np.ndarray | None = None

        if (uris is None) == (paths is None):
            raise ValueError("Exactly one of 'uris' and 'paths' must be provided.")

        super().__init__(uris=uris, paths=paths)

    def _validate_dataset(self) -> None:

        if self.fold is not None:
            if "fold" not in self.slides.column_names:
                raise ValueError("Slides are missing 'fold' column.")
            if self.fold not in self.slides.unique("fold"):
                raise ValueError(f"'{self.fold}' is not a valid fold.")

        for col in self.qc_and_tissue_thresholds:
            if col not in self.tiles.column_names:
                raise ValueError(f"Tiles are missing '{col}' column.")

    def _build_filter_mask(self) -> None:

        table = self.tiles.data.table

        mask = pa.repeat(pa.scalar(True), len(table))
        for column_name, threshold in self.qc_and_tissue_thresholds.items():
            if column_name == "tissue_roi_percentage":
                mask = pc.and_(mask, pc.greater(table[column_name], threshold))
            else:
                mask = pc.and_(mask, pc.less_equal(table[column_name], threshold))

        self._qc_and_tissue_mask = mask.to_numpy(zero_copy_only=False)

    def _get_base_filtered_indices(self, slide: Slide) -> np.ndarray:

        if self._qc_and_tissue_mask is None:
            self._build_filter_mask()

        indices = self._slide_id_to_indices.get(slide["id"])

        if indices is None:
            return np.empty(0, dtype=np.int64)

        np_indices = indices.values.to_numpy()

        assert self._qc_and_tissue_mask is not None
        return np_indices[self._qc_and_tissue_mask[np_indices]]

    def _filter_slides_by_fold(self) -> HFDataset:
        if self.fold is None:
            return self.slides
        return (
            self.slides.filter(lambda s: s["fold"] != self.fold)
            if self.invert_fold_selection
            else self.slides.filter(lambda s: s["fold"] == self.fold)
        )

    @abstractmethod
    def _filter_tiles_by_slide_and_thresholds(self, slide: Slide) -> TileDataset[T_co]:
        pass

    def generate_datasets(self) -> Iterable[TileDataset[T_co]]:

        self._validate_dataset()

        for slide in self._filter_slides_by_fold():
            tile_dataset = self._filter_tiles_by_slide_and_thresholds(slide)

            if len(tile_dataset) == 0:
                print(
                    f"Warning: slide {slide['stem']} has no tiles "
                    f"left after filtering and it will be skipped."
                )
                continue

            yield tile_dataset


class UnlabeledSlideDataset(SlideDataset[T_co], ABC):
    @abstractmethod
    def _generate_tile_dataset(
        self,
        slide: Slide,
        tiles: Tiles,
    ) -> UnlabeledTileDataset[T_co]:
        pass

    def _filter_tiles_by_slide_and_thresholds(
        self, slide: Slide
    ) -> UnlabeledTileDataset[T_co]:
        indices = self._get_base_filtered_indices(slide)
        return self._generate_tile_dataset(slide, Tiles(self.tiles, indices))


class LabeledSlideDataset(SlideDataset[T_co], ABC):
    def __init__(
        self,
        labels_map: dict[str, int],
        apply_carcinoma_prediction_filter: bool,
        carcinoma_prediction_threshold: float | None,
        qc_and_tissue_thresholds: dict[str, float],
        fold: int | None = None,
        invert_fold_selection: bool = False,
        uris: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
    ) -> None:

        if labels_map.get("None") != 0:
            raise ValueError("Label of negative slides is expected to be 0.")

        self.labels_map = labels_map

        if apply_carcinoma_prediction_filter and carcinoma_prediction_threshold is None:
            raise ValueError(
                "Unable to apply carcinoma prediction filter. The "
                "carcinoma prediction threshold must be specified."
            )

        self.carcinoma_prediction_threshold = carcinoma_prediction_threshold
        self.apply_carcinoma_prediction_filter = apply_carcinoma_prediction_filter
        self._carcinoma_prediction_mask: np.ndarray | None = None

        super().__init__(
            qc_and_tissue_thresholds=qc_and_tissue_thresholds,
            fold=fold,
            invert_fold_selection=invert_fold_selection,
            uris=uris,
            paths=paths,
        )

    @override
    def _validate_dataset(self) -> None:

        super()._validate_dataset()

        for col in ("gleason_score", "carcinoma"):
            if col not in self.slides.column_names:
                raise ValueError(f"Slides are missing '{col}' column.")

        if (
            self.carcinoma_prediction_threshold is not None
            and "prediction" not in self.tiles.column_names
        ):
            raise ValueError("Tiles are missing 'prediction' column.")

        assert self.labels_map is not None

        expected_labels = set(self.labels_map.keys())
        found_labels = set(self.slides.unique("gleason_score"))
        unknown_labels = found_labels - expected_labels

        if len(unknown_labels) > 0:
            raise ValueError(
                f"Dataset contains unexpected gleason score labels. Expected "
                f"labels: {expected_labels}. Unknown labels: {unknown_labels}"
            )

    @override
    def _build_filter_mask(self) -> None:

        super()._build_filter_mask()

        if self.carcinoma_prediction_threshold is not None:
            self._carcinoma_prediction_mask = pc.greater(
                self.tiles.data.table["prediction"],
                self.carcinoma_prediction_threshold,
            ).to_numpy(zero_copy_only=False)

    @abstractmethod
    def _generate_tile_dataset(
        self,
        slide: Slide,
        tiles: Tiles,
        slide_label: torch.Tensor,
        tile_labels: torch.Tensor,
    ) -> LabeledTileDataset[T_co]:
        pass

    def _filter_tiles_by_slide_and_thresholds(
        self, slide: Slide
    ) -> LabeledTileDataset[T_co]:

        indices = self._get_base_filtered_indices(slide)

        slide_label = torch.tensor(
            self.labels_map[slide["gleason_score"]],
            dtype=torch.long,
        )

        if slide["carcinoma"] and self._carcinoma_prediction_mask is not None:
            binary_labels = self._carcinoma_prediction_mask[indices]

            if self.apply_carcinoma_prediction_filter:
                indices = indices[binary_labels]
                tile_labels = torch.full(
                    (len(indices),), slide_label.item(), dtype=torch.long
                )
            else:
                tile_labels = torch.where(
                    torch.from_numpy(binary_labels),
                    slide_label,
                    torch.tensor(0, dtype=torch.long),
                )
        else:
            tile_labels = torch.full(
                (len(indices),), slide_label.item(), dtype=torch.long
            )

        return self._generate_tile_dataset(
            slide,
            Tiles(self.tiles, indices),
            slide_label,
            tile_labels,
        )

    def get_slide_labels(self) -> dict[str, torch.Tensor]:

        labels: dict[str, torch.Tensor] = {}

        for dataset in self.datasets:
            assert isinstance(dataset, LabeledTileDataset)
            labels[dataset.slide["stem"]] = dataset.slide_label

        return labels

    def get_tile_labels(self) -> torch.Tensor:
        labels: list[torch.Tensor] = []

        for dataset in self.datasets:
            assert isinstance(dataset, LabeledTileDataset)
            labels.append(dataset.tile_labels)

        return torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)
