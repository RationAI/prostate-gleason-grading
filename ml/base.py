from abc import ABC, abstractmethod

from lightning import LightningModule
from lightning.pytorch import loggers
from matplotlib import pyplot as plt
from torch import Tensor, nn, softmax
from torch.optim.optimizer import Optimizer
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassCohenKappa,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassNegativePredictiveValue,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)

from ml.typing import LabeledSampleBatch, UnlabeledSampleBatch


def metrics(
    prefix: str, num_classes: int, argmax_only: bool = False
) -> MetricCollection:

    metrics_dict: dict[str, Metric | MetricCollection] = {
        "ACC": MulticlassAccuracy(num_classes),
        "QWK": MulticlassCohenKappa(num_classes, weights="quadratic"),
        "F1": MulticlassF1Score(num_classes),
        "Sensitivity": MulticlassRecall(num_classes),
        "Specificity": MulticlassSpecificity(num_classes),
        "PPV": MulticlassPrecision(num_classes),
        "NPV": MulticlassNegativePredictiveValue(num_classes),
    }

    if not argmax_only:
        metrics_dict["AUROC"] = MulticlassAUROC(num_classes)

    return MetricCollection(metrics_dict, prefix=prefix)


class GleasonModel(ABC, LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

        self.train_metrics = metrics("train/", num_classes)
        self.val_metrics = metrics("val/", num_classes)
        self.test_metrics = metrics("test/", num_classes)

        self.train_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        self.val_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        pass

    def _logits_to_prob(self, logits: Tensor) -> Tensor:
        return softmax(logits, dim=1)

    def _log_metrics(self, metrics: MetricCollection) -> None:
        computed = metrics.compute()
        self.log_dict(computed, on_epoch=True, on_step=False)
        metrics.reset()

    def _log_confusion_matrix(self, cm: MulticlassConfusionMatrix, stage: str) -> None:
        confusion_matrix = cm.compute()

        if (
            self.global_rank == 0
            and self.logger is not None
            and isinstance(self.logger, loggers.MLFlowLogger)
        ):
            fig, _ = cm.plot(val=confusion_matrix, add_text=True, cmap="Reds")

            try:
                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    fig,
                    f"confusion_matrix/{stage}_epoch_{self.current_epoch}.png",
                )
            finally:
                plt.close(fig)

        cm.reset()

    def training_step(self, batch: LabeledSampleBatch, batch_idx: int) -> Tensor | None:
        inputs, _, targets = batch
        logits = self(inputs)

        loss = self.criterion(logits, targets)
        self.log(
            "train/loss",
            loss,
            batch_size=len(inputs),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_cm.update(logits, targets)
        self.train_metrics.update(logits, targets)

        return loss

    def validation_step(self, batch: LabeledSampleBatch) -> None:
        inputs, _, targets = batch
        logits = self(inputs)

        loss = self.criterion(logits, targets)
        self.log(
            "validation/loss",
            loss,
            batch_size=len(inputs),
            on_epoch=True,
            prog_bar=True,
        )

        self.val_cm.update(logits, targets)
        self.val_metrics.update(logits, targets)

    def test_step(
        self, batch: LabeledSampleBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        inputs, _, targets = batch
        logits = self(inputs)

        self.test_cm.update(logits, targets)
        self.test_metrics.update(logits, targets)

        return self._logits_to_prob(logits).detach()

    def predict_step(
        self, batch: UnlabeledSampleBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        inputs, _ = batch
        return self._logits_to_prob(self(inputs)).detach()

    def on_train_epoch_end(self) -> None:
        self._log_metrics(self.train_metrics)
        self._log_confusion_matrix(self.train_cm, "train")

    def on_validation_epoch_end(self) -> None:
        self._log_metrics(self.val_metrics)
        self._log_confusion_matrix(self.val_cm, "val")

    def on_test_epoch_end(self) -> None:
        self._log_metrics(self.test_metrics)
        self._log_confusion_matrix(self.test_cm, "test")
