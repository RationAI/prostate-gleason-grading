from typing import TYPE_CHECKING, Any, cast

import torch
from lightning import LightningModule, Trainer
from rationai.mlkit.lightning.callbacks import MultiloaderLifecycle
from rationai.mlkit.lightning.loggers import MLFlowLogger

from ml.typing import UnlabeledSampleBatch


if TYPE_CHECKING:
    from ml.base import GleasonModel
    from ml.datamodule import DataModule
    from ml.datamodule.datasets.base import TileDataset


class PredictionTableCallback(MultiloaderLifecycle):
    def __init__(
        self,
        table_path: str = "prediction_table",
        output_class_names: list[str] | None = None,
    ) -> None:

        super().__init__()

        self.table: dict[str, Any]
        self.slide: str

        self.table_path = table_path
        self.output_class_names = (
            output_class_names
            if output_class_names is not None
            else ["P(negative)", "P(GP3)", "P(GP4+)"]
        )

    def on_predict_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        output_classes = cast("GleasonModel", pl_module).num_classes
        if len(self.output_class_names) != output_classes:
            raise ValueError(
                "The number of model output classes does"
                " not match the number of provided names."
            )

    def on_predict_dataloader_start(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:

        datamodule: DataModule = cast("Any", trainer).datamodule
        dataset = datamodule.predict.datasets[dataloader_idx]
        self.slide = cast("TileDataset[Any]", dataset).slide["stem"]

        self.table = {
            "slide": [],
            "x": [],
            "y": [],
            **{cls: [] for cls in self.output_class_names},
        }

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
        x = metadata["x"].tolist()
        y = metadata["y"].tolist()
        probs = outputs.detach().cpu()
        batch_size = len(probs)

        assert batch_size == len(x) and batch_size == len(y)

        self.table["slide"].extend([self.slide] * batch_size)
        self.table["x"].extend(x)
        self.table["y"].extend(y)

        for i, cls in enumerate(self.output_class_names):
            self.table[cls].extend(probs[:, i].tolist())

    def on_predict_dataloader_end(
        self, trainer: Trainer, pl_module: LightningModule, dataloader_idx: int
    ) -> None:
        logger = cast("MLFlowLogger", trainer.logger)
        logger.log_table(self.table, f"{self.table_path}.json")
