from collections.abc import Iterable
from typing import Any, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset as HFDataset
from rationai.mlkit.data.datasets import MetaTiledSlides


T = TypeVar("T", covariant=True)

type MaskType = pa.BooleanArray | pa.ChunkedArray


class FilterableDataset(MetaTiledSlides[T]):
    def __init__(
        self,
        uris: Iterable[str],
        qc_and_tissue_thresholds: dict[str, float],
        carcinoma_prediction_threshold: float | None,
    ) -> None:

        self.labeled = carcinoma_prediction_threshold is not None
        self.qc_and_tissue_thresholds = qc_and_tissue_thresholds
        self.carcinoma_prediction_threshold = carcinoma_prediction_threshold

        self._qc_and_tissue_mask: MaskType | None = None
        self._carcinoma_prediction_mask: MaskType | None = None

        self.slides: HFDataset
        self.tiles: HFDataset

        super().__init__(uris=uris)

    def _build_filter_masks(self) -> None:

        if self.labeled and (
            "gleason_score" not in self.slides.column_names
            or "carcinoma" not in self.slides.column_names
            or "prediction" not in self.tiles.column_names
        ):
            raise ValueError(
                "Dataset is expected to be labeled but no labels were found."
            )

        table = self.tiles.data.table

        mask = pa.repeat(True, len(table))

        for column_name, threshold in self.qc_and_tissue_thresholds.items():
            if column_name == "tissue_roi_percentage":
                mask = pc.and_(mask, pc.greater(table[column_name], threshold))
            elif column_name in table.column_names:
                mask = pc.and_(mask, pc.less(table[column_name], threshold))
            else:
                raise ValueError(f"Unknown threshold: {column_name}")

        self._qc_and_tissue_mask = mask

        if self.labeled:
            carcinoma_prediction_mask = pc.greater(
                table["prediction"],
                self.carcinoma_prediction_threshold,
            )
            self._carcinoma_prediction_mask = pc.and_(
                mask,
                carcinoma_prediction_mask,
            )

    def indices_of_filtered_tiles(self, slide: dict[str, Any]) -> np.ndarray:

        if self._qc_and_tissue_mask is None:
            self._build_filter_masks()

        mask = (
            self._carcinoma_prediction_mask
            if self.labeled and slide["carcinoma"]
            else self._qc_and_tissue_mask
        )

        assert mask is not None

        tiles_range = self._slide_id_to_indices.get(slide["id"], range(0))
        return pc.indices_nonzero(mask[tiles_range.start : tiles_range.stop]).to_numpy()
