from abc import abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import Dataset


T_co = TypeVar("T_co", covariant=True)


class SlideTiles:
    def __init__(
        self,
        slide: dict[str, Any],
        tiles: HFDataset,
        tile_indices: np.ndarray,
        slide_label: torch.Tensor | None = None,
        tile_labels: torch.Tensor | None = None,
    ) -> None:

        if slide_label is not None and slide_label.ndim != 0:
            raise ValueError("Scalar tensor is expected as a slide label.")

        if tile_labels is not None and len(tile_indices) != tile_labels.numel():
            raise ValueError(
                "The number of tile labels must match the number of tiles."
            )

        self.slide = slide
        self._tiles = tiles
        self._index_map = tile_indices
        self.slide_label = slide_label
        self.tile_labels = tile_labels

    def __len__(self) -> int:
        return len(self._index_map)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._tiles[self._index_map[idx]]


class FilterableDataset(MetaTiledSlides[T_co]):
    def __init__(
        self,
        labeled: bool,
        qc_and_tissue_thresholds: dict[str, float],
        carcinoma_prediction_threshold: float | None,
        apply_carcinoma_prediction_filter: bool,
        uris: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
        fold: int | None = None,
        invert_fold_selection: bool = False,
        labels_map: dict[str, int] | None = None,
    ) -> None:

        if labeled:
            if labels_map is None:
                raise ValueError("Labels map is expected for labeled dataset.")
            if labels_map.get("None") != 0:
                raise ValueError("Label of negative slides is expected to be 0.")

        self.labeled = labeled
        self.labels_map = labels_map

        self.fold = fold
        self.invert_fold_selection = invert_fold_selection

        if apply_carcinoma_prediction_filter and (
            not labeled or carcinoma_prediction_threshold is None
        ):
            raise ValueError(
                "Unable to apply carcinoma prediction filter. The "
                "dataset must be labeled and threshold must be specified."
            )

        self.qc_and_tissue_thresholds = qc_and_tissue_thresholds
        self.carcinoma_prediction_threshold = carcinoma_prediction_threshold
        self.apply_carcinoma_prediction_filter = apply_carcinoma_prediction_filter

        self._qc_and_tissue_mask: np.ndarray | None = None
        self._carcinoma_prediction_mask: np.ndarray | None = None

        self.slides: HFDataset
        self.tiles: HFDataset

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

        if not self.labeled:
            return

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

    def _build_filter_masks(self) -> None:

        table = self.tiles.data.table

        mask = pa.repeat(pa.scalar(True), len(table))

        for column_name, threshold in self.qc_and_tissue_thresholds.items():
            if column_name == "tissue_roi_percentage":
                mask = pc.and_(mask, pc.greater(table[column_name], threshold))
            else:
                mask = pc.and_(mask, pc.less_equal(table[column_name], threshold))

        self._qc_and_tissue_mask = mask.to_numpy(zero_copy_only=False)

        if self.labeled and self.carcinoma_prediction_threshold is not None:
            mask = pc.and_(
                mask,
                pc.greater(table["prediction"], self.carcinoma_prediction_threshold),
            )
            self._carcinoma_prediction_mask = mask.to_numpy(zero_copy_only=False)

    def _filter_slides_by_fold(self) -> HFDataset:
        if self.fold is None:
            return self.slides
        return (
            self.slides.filter(lambda s: s["fold"] != self.fold)
            if self.invert_fold_selection
            else self.slides.filter(lambda s: s["fold"] == self.fold)
        )

    def _filter_tiles_by_slide_and_thresholds(
        self, slide: dict[str, Any]
    ) -> SlideTiles:

        indices = self._slide_id_to_indices.get(slide["id"])

        if indices is None:
            return SlideTiles(slide, self.tiles, np.empty(0, dtype=np.int64))

        np_indices = indices.values.to_numpy()

        if self._qc_and_tissue_mask is None:
            self._build_filter_masks()

        assert self._qc_and_tissue_mask is not None
        np_indices = np_indices[self._qc_and_tissue_mask[np_indices]]

        if not self.labeled:
            return SlideTiles(slide, self.tiles, np_indices)

        assert self.labels_map is not None
        slide_label = torch.tensor(
            self.labels_map[slide["gleason_score"]],
            dtype=torch.long,
        )

        if slide["carcinoma"] and self._carcinoma_prediction_mask is not None:
            binary_labels = self._carcinoma_prediction_mask[np_indices]

            if self.apply_carcinoma_prediction_filter:
                np_indices = np_indices[binary_labels]
                tile_labels = torch.full(
                    (len(np_indices),), slide_label.item(), dtype=torch.long
                )
            else:
                tile_labels = torch.where(
                    torch.from_numpy(binary_labels),
                    slide_label,
                    torch.tensor(0, dtype=torch.long),
                )
        else:
            tile_labels = torch.full(
                (len(np_indices),), slide_label.item(), dtype=torch.long
            )

        return SlideTiles(slide, self.tiles, np_indices, slide_label, tile_labels)

    def generate_datasets(self) -> Iterable[Dataset[T_co]]:

        self._validate_dataset()

        for slide in self._filter_slides_by_fold():
            slide_tiles = self._filter_tiles_by_slide_and_thresholds(slide)

            if len(slide_tiles) == 0:
                print(
                    f"Warning: slide {slide['stem']} has no tiles "
                    f"left after filtering and it will be skipped."
                )
                continue

            yield self._generate_slide_dataset(slide_tiles)

    @abstractmethod
    def _generate_slide_dataset(self, slide_tiles: SlideTiles) -> Dataset[T_co]:
        pass
