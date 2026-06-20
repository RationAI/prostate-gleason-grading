from collections.abc import Iterable
from typing import Literal, cast, overload

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import DataLoader

from ml.typing import (
    LabeledSample,
    LabeledSampleBatch,
    UnlabeledSample,
    UnlabeledSampleBatch,
)


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        drop_last: bool = True,
        shuffle: bool = True,
        validation_fold: int | None = None,
        sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:

        super().__init__()

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler

        self.fold = validation_fold
        self.datasets = datasets

        self.num_workers = num_workers

    @overload
    def _instantiate_dataset(
        self, mode: Literal["train", "val", "test"]
    ) -> MetaTiledSlides[LabeledSample]: ...

    @overload
    def _instantiate_dataset(
        self, mode: Literal["predict"]
    ) -> MetaTiledSlides[UnlabeledSample]: ...

    def _instantiate_dataset(
        self, mode: str
    ) -> MetaTiledSlides[LabeledSample] | MetaTiledSlides[UnlabeledSample]:

        fold = self.fold if mode in {"train", "val"} else None
        dataset = instantiate(self.datasets[mode], fold=fold, mode=mode)

        return (
            cast("MetaTiledSlides[UnlabeledSample]", dataset)
            if mode == "predict"
            else cast("MetaTiledSlides[LabeledSample]", dataset)
        )

    def setup(self, stage: str) -> None:
        match stage:
            case "fit":
                self.train = self._instantiate_dataset("train")
                self.val = self._instantiate_dataset("val")
            case "validate":
                self.val = self._instantiate_dataset("val")
            case "test":
                self.test = self._instantiate_dataset("test")
            case "predict":
                self.predict = self._instantiate_dataset("predict")

    def get_train_labels(self) -> torch.Tensor:
        return torch.tensor(
            [label.item() for _, _, label in self.train], dtype=torch.long
        )

    def train_dataloader(self) -> Iterable[LabeledSampleBatch]:
        sampler = (
            instantiate(self.sampler, labels=self.get_train_labels())
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
