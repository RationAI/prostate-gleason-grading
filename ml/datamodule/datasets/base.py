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


class FilterableDataset(MetaTiledSlides[T_co]):
    def __init__(
        self,
        qc_and_tissue_thresholds: dict[str, float],
        carcinoma_prediction_threshold: float | None,
        uris: Iterable[str] | None = None,
        paths: Iterable[Path | str] | None = None,
        mode: str | None = None,
        fold: int | None = None,
        invert_fold_selection: bool = False,
        labels_map: dict[str, int] | None = None,
    ) -> None:

        self.labeled = carcinoma_prediction_threshold is not None
        self.qc_and_tissue_thresholds = qc_and_tissue_thresholds
        self.prediction_threshold = carcinoma_prediction_threshold
        self.labels_map = labels_map

        if self.labeled and self.labels_map is None:
            raise ValueError("Labels map is expected for labeled dataset.")

        self.mode = mode
        self.fold = fold
        self.invert_fold_selection = invert_fold_selection

        if self.fold is not None and self.mode not in {"train", "val"}:
            raise ValueError(
                f"Invalid mode '{self.mode}': if fold is specified, "
                f"mode must be one of 'train' or 'val'"
            )

        if self.mode in {"train", "val", "test"} and not self.labeled:
            raise ValueError(f"The dataset must be labeled when mode is '{self.mode}'")

        self._qc_and_tissue_mask: np.ndarray | None = None
        self._carcinoma_prediction_mask: np.ndarray | None = None

        self.slides: HFDataset
        self.tiles: HFDataset

        if (uris is None) == (paths is None):
            raise ValueError("Exactly one of 'uris' and 'paths' must be provided.")

        super().__init__(uris=uris, paths=paths)

    def _check_labels(self) -> None:

        if (
            "gleason_score" not in self.slides.column_names
            or "carcinoma" not in self.slides.column_names
            or "prediction" not in self.tiles.column_names
        ):
            raise ValueError(
                "Dataset is expected to be labeled but no labels were found."
            )

        assert self.labels_map is not None

        expected_labels = set(self.labels_map.keys())
        found_labels = set(self.slides.unique("gleason_score"))
        unknown_labels = found_labels - expected_labels

        if len(unknown_labels) > 0:
            raise ValueError(
                f"Unknown labels: {unknown_labels}. Expected labels: {expected_labels}."
            )

    def _build_filter_masks(self) -> None:

        table = self.tiles.data.table

        mask = pa.repeat(pa.scalar(True), len(table))

        for column_name, threshold in self.qc_and_tissue_thresholds.items():
            if column_name not in table.column_names:
                raise ValueError(
                    f"Threshold column '{column_name}' not found in tiles"
                    f" table. Available columns: {table.column_names}"
                )
            if column_name == "tissue_roi_percentage":
                mask = pc.and_(mask, pc.greater(table[column_name], threshold))
            else:
                mask = pc.and_(mask, pc.less_equal(table[column_name], threshold))

        self._qc_and_tissue_mask = mask.to_numpy(zero_copy_only=False)

        if self.labeled:
            mask = pc.and_(
                mask, pc.greater(table["prediction"], self.prediction_threshold)
            )
            self._carcinoma_prediction_mask = mask.to_numpy(zero_copy_only=False)

    def _filter_slides_by_fold(self) -> HFDataset:

        if self.fold is not None:
            if "fold" not in self.slides.column_names:
                raise ValueError("Fold filtering requires a 'fold' column in slides.")
            if self.fold not in self.slides.unique("fold"):
                raise ValueError(f"Unknown fold: {self.fold}")

            assert self.mode in {"train", "val"}

            if (self.mode == "train") ^ self.invert_fold_selection:
                return self.slides.filter(lambda s: s["fold"] != self.fold)
            else:
                return self.slides.filter(lambda s: s["fold"] == self.fold)

        return self.slides

    def _filter_tiles_by_slide_and_thresholds(self, slide: dict[str, Any]) -> HFDataset:

        indices = self._slide_id_to_indices.get(slide["id"])

        if indices is None:
            return self.tiles.select([])

        np_indices = indices.values.to_numpy()

        if self._qc_and_tissue_mask is None:
            self._build_filter_masks()

        mask = (
            self._carcinoma_prediction_mask
            if self.labeled and slide["carcinoma"]
            else self._qc_and_tissue_mask
        )

        assert mask is not None

        np_indices = np_indices[mask[np_indices]]
        return self.tiles.select(np_indices)

    def generate_datasets(self) -> Iterable[Dataset[T_co]]:

        if self.labeled:
            self._check_labels()

        for slide in self._filter_slides_by_fold():
            label = None

            if self.labeled:
                assert self.labels_map is not None
                label = torch.tensor(
                    self.labels_map[slide["gleason_score"]],
                    dtype=torch.long,
                )

            tiles = self._filter_tiles_by_slide_and_thresholds(slide)

            if len(tiles) == 0:
                print(
                    f"Warning: slide {slide['stem']} has no tiles "
                    f"left after filtering - it will be skipped"
                )
                continue

            yield self._generate_slide_dataset(slide, tiles, label)

    @abstractmethod
    def _generate_slide_dataset(
        self,
        slide: dict[str, Any],
        tiles: HFDataset,
        label: torch.Tensor | None,
    ) -> Dataset[T_co]:
        pass
