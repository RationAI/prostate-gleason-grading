from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassConfusionMatrix

from ml.aggregators import BinaryAggregator, GleasonScoreAggregator
from ml.base import metrics


if TYPE_CHECKING:
    from ml.base import TLGleasonModel
    from ml.datamodule.datasets.base import FullyLabeledTileDataset
    from ml.datamodule.samples_datamodule import SamplesDataModule


class AggregationCallback(MultiloaderLifecycle):
    def __init__(
        self,
        aggregators: dict[str, GleasonScoreAggregator | BinaryAggregator],
        table_path: str = "SL_prediction_table",
    ) -> None:

        super().__init__()

        self.aggregators = aggregators
        self.table_path = table_path + ".json"

        self.buffer: dict[str, list[Tensor]] = {"probs": [], "x": [], "y": []}

        self.metrics: dict[str, MetricCollection] = {}
        self.cms: dict[str, MulticlassConfusionMatrix] = {}

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for name, aggr in self.aggregators.items():
            if isinstance(aggr, BinaryAggregator):
                continue

            num_classes = aggr.num_classes()

            self.cms[name] = MulticlassConfusionMatrix(num_classes)
            self.cms[name] = self.cms[name].to(pl_module.device)

            self.metrics[name] = metrics(f"SL/{name}/", num_classes, True)
            self.metrics[name] = self.metrics[name].to(pl_module.device)

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
        metadata = outputs["metadata"]
        self.buffer["probs"].append(outputs["prob"].cpu())
        self.buffer["x"].append(metadata["x"].cpu())
        self.buffer["y"].append(metadata["y"].cpu())

    def on_test_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:

        datamodule: SamplesDataModule = cast("Any", trainer).datamodule
        ds = cast("FullyLabeledTileDataset", datamodule.test.datasets[dataloader_idx])

        probs = torch.cat(self.buffer["probs"], dim=0)
        x = torch.cat(self.buffer["x"], dim=0)
        y = torch.cat(self.buffer["y"], dim=0)

        self.buffer = {"probs": [], "x": [], "y": []}

        table_row = {
            "slide": ds.slide["stem"],
            "label": ds.slide["gleason_score"],
        }

        label = ds.slide_label.to(pl_module.device)

        for name, aggr in self.aggregators.items():
            if isinstance(aggr, BinaryAggregator):
                table_row[name] = aggr(probs, x, y, ds.slide)
                continue

            gs_tensor, gs_str = aggr(probs, x, y, ds.slide)
            gs_tensor = gs_tensor.to(pl_module.device)

            table_row[name] = gs_str

            self.metrics[name].update(gs_tensor, label)
            self.cms[name].update(gs_tensor, label)

        logger = cast("MLFlowLogger", trainer.logger)
        logger.log_table(table_row, self.table_path)

    def on_test_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        module = cast("TLGleasonModel", pl_module)
        for _metrics in self.metrics.values():
            module._log_metrics(_metrics)
        for name, cm in self.cms.items():
            module._log_confusion_matrix(cm, f"SL/{name}")
