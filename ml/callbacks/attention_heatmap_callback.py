from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder
from ratiopath.masks.mask_builders.aggregation import MeanAggregator


if TYPE_CHECKING:
    from ml.datamodule.bags_datamodule import BagsDataModule
    from ml.datamodule.datasets.base import BagOfTilesDataset


def min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    t_min, t_max = tensor.min(), tensor.max()
    return (tensor - t_min) / (t_max - t_min)


class AttentionHeatmapCallback(MultiloaderLifecycle):
    def __init__(
        self,
        save_artifact_path: str = "heatmaps",
        save_dir: str | None = None,
    ) -> None:

        super().__init__()

        self.save_artifact_path = save_artifact_path
        self.save_dir = save_dir

        self._slides: dict[str, dict[str, Any]]
        self._logger: MLFlowLogger

    def _on_stage_start(self, trainer: Trainer, stage: str) -> None:
        datamodule: BagsDataModule[Any, Any] = cast("Any", trainer).datamodule
        dataset: BagOfTilesDataset[Any] = getattr(datamodule, stage)
        self._slides = {bag.slide["stem"]: bag.slide for bag in dataset.bags}
        self._logger = cast("MLFlowLogger", trainer.logger)

    def on_predict_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._on_stage_start(trainer, "predict")

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._on_stage_start(trainer, "test")

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self._process_batch(outputs)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        assert isinstance(outputs, dict)
        self._process_batch(outputs)

    def _process_batch(self, outputs: dict[str, Any]) -> None:

        for i, metadata in enumerate(outputs["metadata"]):
            slide = self._slides[metadata["slide"]]

            y = metadata["y"].cpu().numpy()
            x = metadata["x"].cpu().numpy()
            coords = np.stack([y, x], axis=-1)

            attention = outputs["attention"][i]
            attention = attention[: coords.shape[0]]  # remove padding

            # normalize so that max value is 1, min value is 0,
            # otherwise the values are so small, that trying to
            # rescale into 0-255 by multiplying by 255 produces
            # values < 1 and converting to uint8 results in all zeros

            attention = min_max_normalize(attention)
            attention = attention.detach().unsqueeze(-1).cpu().numpy()

            self._create_mask(slide, "attention", attention, coords)

            tile_logits = outputs["tile_logits"][i]
            tile_logits = tile_logits[: coords.shape[0]]  # remove padding

            num_classes = tile_logits.shape[1]

            tile_probs = torch.softmax(tile_logits, dim=-1)

            for cls in range(num_classes):
                class_probs = tile_probs[:, cls].detach().unsqueeze(-1).cpu().numpy()
                self._create_mask(slide, f"class_{cls}", class_probs, coords)

    def _create_mask(
        self,
        slide: dict[str, Any],
        name: str,
        inputs: np.ndarray,
        coords: np.ndarray,
    ) -> None:

        builder = MaskBuilder(
            source_extents=(slide["extent_y"], slide["extent_x"]),
            source_tile_extent=slide["tile_extent_x"],
            output_tile_extent=1,
            stride=slide["stride_x"],
            n_channels=1,
            storage="inmemory",
            aggregation=MeanAggregator,
        )

        builder.update_batch(inputs, coords)

        mask = builder.finalize()["mask"]
        mask = (mask * 255).clip(0, 255).astype(np.uint8)

        mask_vips = builder.resize_to_source(mask, kernel="nearest")
        mask_name = f"{name}/{slide['stem']}.tiff"
        mppx, mppy = slide["mpp_x"], slide["mpp_y"]

        if self.save_dir is not None:
            mask_path = Path(self.save_dir) / mask_name
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            write_big_tiff(mask_vips, mask_path, mppx, mppy)
            self._logger.log_artifact(
                str(mask_path), artifact_path=f"{self.save_artifact_path}/{name}"
            )
        else:
            with TemporaryDirectory() as tmp_dir:
                mask_path = Path(tmp_dir) / mask_name
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                write_big_tiff(mask_vips, mask_path, mppx, mppy)
                self._logger.log_artifact(
                    str(mask_path), artifact_path=f"{self.save_artifact_path}/{name}"
                )

        builder.cleanup()
