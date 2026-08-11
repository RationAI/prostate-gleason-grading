from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np
import pyvips
import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff
from ratiopath.masks.mask_builders import MaskBuilder
from ratiopath.masks.mask_builders.aggregation import MeanAggregator

from ml.typing import LabeledSampleBatch, MetadataBatch, UnlabeledSampleBatch


if TYPE_CHECKING:
    from ml.base import GleasonModel
    from ml.datamodule import DataModule
    from ml.datamodule.datasets.base import TileDataset


class HeatmapCallback(MultiloaderLifecycle):
    def __init__(
        self,
        save_artifact_path: str = "heatmaps",
        save_dir: str | None = None,
    ) -> None:

        super().__init__()

        self.save_artifact_path = save_artifact_path
        self.save_dir = save_dir

        self._slide: dict[str, Any]
        self._mask_builder: MaskBuilder

    def num_channels(self, num_output_classes: int) -> int:
        return num_output_classes

    def _process_batch(
        self,
        probs: np.ndarray,
        coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return probs, coords

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("predict", trainer, pl_module, dataloader_idx)

    def on_test_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_start("test", trainer, pl_module, dataloader_idx)

    def _on_dataloader_start(
        self,
        mode: str,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloader_idx: int,
    ) -> None:

        datamodule: DataModule = cast("Any", trainer).datamodule
        dataset = getattr(datamodule, mode).datasets[dataloader_idx]
        self._slide = cast("TileDataset[Any]", dataset).slide

        module = cast("GleasonModel", pl_module)

        self._mask_builder = MaskBuilder(
            source_extents=(self._slide["extent_y"], self._slide["extent_x"]),
            source_tile_extent=self._slide["tile_extent_x"],
            output_tile_extent=1,
            stride=self._slide["stride_x"],
            n_channels=self.num_channels(module.num_classes),
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
        self._on_batch_end(outputs, metadata)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: LabeledSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, metadata, _ = batch
        assert isinstance(outputs, torch.Tensor)
        self._on_batch_end(outputs, metadata)

    def _on_batch_end(self, outputs: torch.Tensor, metadata: MetadataBatch) -> None:

        probs = outputs.detach().cpu().numpy()

        y = metadata["y"].cpu().numpy()
        x = metadata["x"].cpu().numpy()
        coords = np.stack([y, x], axis=-1)

        probs, coords = self._process_batch(probs, coords)

        if len(probs) > 0:
            self._mask_builder.update_batch(probs, coords)

    def on_test_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_end(trainer)

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_end(trainer)

    def _on_dataloader_end(self, trainer: Trainer) -> None:

        mask_vips = self._prepare_mask()
        mask_name = f"{self._slide['stem']}.tiff"
        mppx, mppy = self._slide["mpp_x"], self._slide["mpp_y"]

        if self.save_dir is not None:
            mask_path = f"{self.save_dir}/{mask_name}"
            write_big_tiff(mask_vips, Path(mask_path), mppx, mppy)

        else:
            with TemporaryDirectory() as tmp_dir:
                mask_path = f"{tmp_dir}/{mask_name}"
                write_big_tiff(mask_vips, Path(mask_path), mppx, mppy)

        logger = cast("MLFlowLogger", trainer.logger)
        logger.log_artifact(mask_path, artifact_path=self.save_artifact_path)

        self._mask_builder.cleanup()

    def _prepare_mask(self) -> pyvips.Image:
        mask = self._mask_builder.finalize()["mask"]
        mask = (mask * 255).clip(0, 255).astype(np.uint8)
        return self._mask_builder.resize_to_source(mask, kernel="nearest")


class RawProbabilityHeatmapCallback(HeatmapCallback): ...


class ConditionalProbabilityHeatmapCallback(HeatmapCallback):
    def __init__(
        self,
        eps: float = 1e-6,
        save_artifact_path: str = "heatmaps",
        save_dir: str | None = None,
    ) -> None:
        super().__init__(save_artifact_path, save_dir)
        if eps <= 0:
            raise ValueError("eps must be greater than 0")
        self.eps = eps

    @override
    def num_channels(self, num_output_classes: int) -> int:
        if num_output_classes != 3:
            raise ValueError(
                "The number of model outputs is expected "
                "to be 3: P(no carcinoma), P(GP3), P(GP4+)"
            )
        return 2

    @override
    def _process_batch(
        self,
        probs: np.ndarray,
        coords: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

        carcinoma_prob = 1 - probs[:, 0]
        bit_mask = carcinoma_prob > self.eps

        probs = probs[bit_mask]
        carcinoma_prob = carcinoma_prob[bit_mask]
        gp4_prob = probs[:, 2] / carcinoma_prob

        return np.stack((carcinoma_prob, gp4_prob), axis=-1), coords[bit_mask]


class ClassifyHeatmapCallback(HeatmapCallback):
    @override
    def _prepare_mask(self) -> pyvips.Image:
        prob_mask = self._mask_builder.finalize()["mask"]

        num_classes = prob_mask.shape[0]
        if num_classes <= 1:
            raise ValueError("Number of classes must be at least two.")

        class_mask = np.expand_dims(np.argmax(prob_mask, axis=0), axis=0)
        class_mask = class_mask / (num_classes - 1)
        class_mask = (class_mask * 255).clip(0, 255).astype(np.uint8)
        return self._mask_builder.resize_to_source(class_mask, kernel="nearest")
