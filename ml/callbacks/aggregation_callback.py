from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix

from ml.aggregators import GleasonScoreAggregator
from ml.base import metrics
from ml.typing import LabeledSampleBatch, MetadataBatch, UnlabeledSampleBatch


if TYPE_CHECKING:
    from ml.base import GleasonModel
    from ml.datamodule.data_module import DataModule
    from ml.datamodule.datasets.base import LabeledTileDataset


class AggregationCallback(MultiloaderLifecycle):
    def __init__(
        self,
        aggregators: dict[str, GleasonScoreAggregator],
        log_table: bool = False,
        table_path: str = "prediction_table",
    ) -> None:

        super().__init__()

        self.aggregators = aggregators
        self.log_table = log_table
        self.table_path = table_path

        self.metrics: dict[str, dict[str, MetricCollection]] = {}
        self.cms: dict[str, dict[str, MulticlassConfusionMatrix]] = {}

        self._buffer: dict[str, list[Tensor]] = {}
        self._reset_buffer()

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: str | None = None,
    ) -> None:

        if stage == "validate":
            stage = "val"
        if stage not in {"val", "test"}:
            return

        self.metrics[stage], self.cms[stage] = {}, {}

        num_tile_classes = cast("GleasonModel", pl_module).num_classes

        for aggr_name, aggr in self.aggregators.items():
            num_slide_classes = aggr.num_output_classes(num_tile_classes)

            self.cms[stage][aggr_name] = MulticlassConfusionMatrix(
                num_slide_classes
            ).to(pl_module.device)

            self.metrics[stage][aggr_name] = metrics(
                f"{stage}/SL/{aggr_name}/", num_slide_classes
            ).to(pl_module.device)

    def _reset_buffer(self) -> None:
        self._buffer = {"probs": [], "x": [], "y": []}

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: LabeledSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 0:
            _, metadata, _ = batch
            assert isinstance(outputs, Mapping)
            self._on_batch_end(outputs["prob"], metadata)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: LabeledSampleBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if dataloader_idx != 0:
            _, metadata, _ = batch
            assert isinstance(outputs, torch.Tensor)
            self._on_batch_end(outputs, metadata)

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

    def _on_batch_end(self, outputs: Tensor, metadata: MetadataBatch) -> None:
        self._buffer["probs"].append(outputs.cpu())
        self._buffer["x"].append(metadata["x"].cpu())
        self._buffer["y"].append(metadata["y"].cpu())

    def on_validation_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        if dataloader_idx != 0:
            self._on_dataloader_end("val", trainer, pl_module, dataloader_idx - 1)

    def on_test_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        if dataloader_idx != 0:
            self._on_dataloader_end("test", trainer, pl_module, dataloader_idx - 1)

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        self._on_dataloader_end("predict", trainer, pl_module, dataloader_idx)

    def _on_dataloader_end(
        self,
        stage: str,
        trainer: Trainer,
        pl_module: LightningModule,
        dataloader_idx: int,
    ) -> None:

        datamodule: DataModule = cast("Any", trainer).datamodule
        ds_name = f"{stage}_sl" if stage != "predict" else stage
        ds = getattr(datamodule, ds_name).datasets[dataloader_idx]

        probs = torch.cat(self._buffer["probs"], dim=0)
        x = torch.cat(self._buffer["x"], dim=0)
        y = torch.cat(self._buffer["y"], dim=0)

        self._reset_buffer()

        table_row = {"slide": ds.slide["stem"]}

        if stage != "predict":
            table_row["target"] = ds.slide["gleason_score"]
            target = cast("LabeledTileDataset", ds).slide_label.to(pl_module.device)

        for aggr_name, aggr in self.aggregators.items():
            gs_tensor, gs_str = aggr(probs, x, y, ds.slide)
            gs_tensor = gs_tensor.to(pl_module.device)

            table_row[f"{aggr_name}_prediction"] = gs_str

            if stage != "predict":
                self.metrics[stage][aggr_name].update(gs_tensor, target)
                self.cms[stage][aggr_name].update(gs_tensor, target)

        if self.log_table:
            logger = cast("MLFlowLogger", trainer.logger)
            logger.log_table(
                table_row,
                f"{self.table_path}/{stage}/.json",
            )

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._on_epoch_end("val", pl_module)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._on_epoch_end("test", pl_module)

    def _on_epoch_end(self, stage: str, pl_module: LightningModule) -> None:
        module = cast("GleasonModel", pl_module)
        for _metrics in self.metrics[stage].values():
            module._log_metrics(_metrics)
        for aggr_name, cm in self.cms[stage].items():
            module._log_confusion_matrix(cm, f"{stage}/SL/{aggr_name}")
