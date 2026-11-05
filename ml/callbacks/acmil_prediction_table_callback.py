from collections.abc import Mapping
from typing import Any, cast

import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torch import Tensor


class ACMILPredictionsTableCallback(MultiloaderLifecycle):
    def __init__(self, table_path: str = "prediction_table") -> None:

        super().__init__()

        self.table_path = table_path
        self.branch_names = ["Negative", "GP3", "GP4", "GP5"]

        self._logger: MLFlowLogger
        self._labels_map: dict[int, str]

    def on_test_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        self._logger = cast("MLFlowLogger", trainer.logger)
        self._labels_map = cast(
            "Any", trainer
        ).datamodule.test.slide_dataset.labels_map_inverse
        if self._labels_map is None:
            raise RuntimeError("Labels are expected to be invertible.")

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

        predictions: Tensor = outputs["pred"]
        targets: Tensor = outputs["targets"]
        branch_predictions: list[Tensor] = outputs["branch_predictions"]

        table: dict[str, list[Any]] = {
            "Slide": [],
            "GS_pred": [],
            "GS_targ": [],
            **{branch_name: [] for branch_name in self.branch_names},
        }

        for i, metadata in enumerate(outputs["metadata"]):
            table["Slide"].append(metadata["slide"])

            prediction = int(predictions[i].item())
            target = int(targets[i].item())

            table["GS_pred"].append(self._labels_map[prediction])
            table["GS_targ"].append(self._labels_map[target])

            for j, branch_prediction in enumerate(branch_predictions):
                table[self.branch_names[j]].append(branch_prediction[i].item())

        self._logger.log_table(table, f"{self.table_path}.json")
