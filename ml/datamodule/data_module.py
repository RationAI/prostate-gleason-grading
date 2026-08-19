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
    predict: UnlabeledSlideDataset
    val_tl: LabeledSlideDataset
    val_sl: LabeledSlideDataset | None
    test_tl: LabeledSlideDataset
    test_sl: LabeledSlideDataset | None

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
                self.val_tl = instantiate(self.datasets["val_tl"])
                self.val_sl = instantiate(self.datasets.get("val_sl"))
            case "validate":
                self.val_tl = instantiate(self.datasets["val_tl"])
                self.val_sl = instantiate(self.datasets.get("val_sl"))
            case "test":
                self.test_tl = instantiate(self.datasets["test_tl"])
                self.test_sl = instantiate(self.datasets.get("test_sl"))
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

    def _get_dataloaders(
        self,
        dataset_tl: LabeledSlideDataset,
        dataset_sl: LabeledSlideDataset | None,
    ) -> list[Iterable[LabeledSampleBatch]]:

        dataloaders: list[Iterable[LabeledSampleBatch]] = [
            DataLoader(
                dataset_tl,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        ]

        if dataset_sl is not None:
            for dataset in dataset_sl.datasets:
                dataloaders.append(DataLoader(dataset, batch_size=self.batch_size))

        return dataloaders

    def val_dataloader(self) -> list[Iterable[LabeledSampleBatch]]:
        return self._get_dataloaders(self.val_tl, self.val_sl)

    def test_dataloader(self) -> list[Iterable[LabeledSampleBatch]]:
        return self._get_dataloaders(self.test_tl, self.test_sl)

    def predict_dataloader(self) -> list[Iterable[UnlabeledSampleBatch]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]
