from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast

import mlflow
import numpy as np
import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder
from ratiopath.masks.mask_builders.aggregation import MeanAggregator

from ml.typing import UnlabeledSampleBatch


if TYPE_CHECKING:
    from ml.base import GleasonModel
    from ml.datamodule import DataModule
    from ml.datamodule.datasets.base import UnlabeledTileDataset


class HeatmapCallback(MultiloaderLifecycle):
    slide: dict[str, Any]
    mask_builder: MaskBuilder

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:

        module = cast("GleasonModel", pl_module)

        datamodule: DataModule = cast("Any", trainer).datamodule
        self.slide = cast(
            "UnlabeledTileDataset", datamodule.predict.datasets[dataloader_idx]
        ).slide

        self.mask_builder: MaskBuilder = MaskBuilder(
            source_extents=(self.slide["extent_y"], self.slide["extent_x"]),
            source_tile_extent=self.slide["tile_extent_x"],
            output_tile_extent=1,
            stride=self.slide["stride_x"],
            n_channels=module.num_classes,
            storage="inmemory",
            aggregation=MeanAggregator,
        )

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor,
        batch: UnlabeledSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        _, metadata = batch

        probs = outputs.detach().cpu().numpy()

        y = metadata["y"].cpu().numpy()
        x = metadata["x"].cpu().numpy()
        coords = np.stack([y, x], axis=-1)

        self.mask_builder.update_batch(probs, coords)

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:

        mask = self.mask_builder.finalize()["mask"]
        vips_mask = self.mask_builder.resize_to_source(mask, kernel="nearest")

        with TemporaryDirectory() as tmp_dir:
            mask_path = f"{tmp_dir}/{self.slide['stem']}.tiff"

            write_big_tiff(
                image=vips_mask,
                path=Path(mask_path),
                mpp_x=self.slide["mpp_x"],
                mpp_y=self.slide["mpp_y"],
            )

            mlflow.log_artifact(mask_path, artifact_path="heatmaps")

        self.mask_builder.cleanup()
