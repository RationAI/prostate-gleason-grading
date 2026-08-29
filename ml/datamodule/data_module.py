from collections.abc import Iterable

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ml.datamodule.datasets.base import (
    LabeledSlideDataset,
    UnlabeledSlideDataset,
)
from ml.typing import (
    LabeledSampleBatch,
    UnlabeledSampleBatch,
)


class DataModule(LightningDataModule):
    train: LabeledSlideDataset
    val: LabeledSlideDataset
    test: LabeledSlideDataset
    predict: UnlabeledSlideDataset

    def __init__(
        self,
        batch_size: int,
        drop_last: bool,
        shuffle: bool,
        sampler: DictConfig | None = None,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:

        super().__init__()

        self.datasets = datasets
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = instantiate(self.datasets["train"])
                self.val = instantiate(self.datasets["val"])
            case "validate":
                self.val = instantiate(self.datasets["val"])
            case "test":
                self.test = instantiate(self.datasets["test"])
            case "predict":
                self.predict = instantiate(self.datasets["predict"])

    def train_dataloader(self) -> Iterable[LabeledSampleBatch]:
        sampler = (
            instantiate(self.sampler, labels=self.train.get_tile_labels())
            if self.sampler is not None
            else None
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None and self.shuffle,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[LabeledSampleBatch]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> list[Iterable[LabeledSampleBatch]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.test.datasets
        ]

    def predict_dataloader(self) -> list[Iterable[UnlabeledSampleBatch]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]
