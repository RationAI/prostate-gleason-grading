from abc import ABC, abstractmethod

from lightning import LightningModule
from torch import Tensor, nn, softmax
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassNegativePredictiveValue,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassSpecificity,
)

from ml.typing import LabeledSampleBatch, UnlabeledSampleBatch


class GleasonModel(ABC, LightningModule):
    def __init__(self, num_classes: int, lr: float) -> None:
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()

        metrics: MetricCollection = MetricCollection(
            {
                "AUROC": MulticlassAUROC(num_classes=num_classes),
                "accuracy": MulticlassAccuracy(num_classes=num_classes),
                "precision": MulticlassPrecision(num_classes=num_classes),
                "recall": MulticlassRecall(num_classes=num_classes),
                "f1": MulticlassF1Score(num_classes=num_classes),
                "specificity": MulticlassSpecificity(num_classes=num_classes),
                "negative_predictive_value": MulticlassNegativePredictiveValue(
                    num_classes=num_classes
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="validation/")
        self.test_metrics = metrics.clone(prefix="test/")

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    def _logits_to_prob(self, logits: Tensor) -> Tensor:
        return softmax(logits, dim=1)

    def training_step(self, batch: LabeledSampleBatch) -> Tensor | None:
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

        self.train_metrics.update(logits, targets)
        self.log_dict(
            self.train_metrics, batch_size=len(inputs), on_step=False, on_epoch=True
        )

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

        self.val_metrics.update(logits, targets)
        self.log_dict(self.val_metrics, batch_size=len(inputs), on_epoch=True)

    def test_step(
        self, batch: LabeledSampleBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        inputs, _, targets = batch
        logits = self(inputs)

        self.test_metrics.update(logits, targets)
        self.log_dict(self.test_metrics, batch_size=len(inputs), on_epoch=True)

        return self._logits_to_prob(logits)

    def predict_step(
        self, batch: UnlabeledSampleBatch, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        inputs, _ = batch
        return self._logits_to_prob(self(inputs))

    def configure_optimizers(self) -> Optimizer:
        return AdamW(self.parameters(), self.lr)
