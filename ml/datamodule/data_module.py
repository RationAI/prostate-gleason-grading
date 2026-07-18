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
        drop_last: bool,
        shuffle: bool,
        sampler: DictConfig | None = None,
        fold: int | None = None,
        invert_fold_selection: bool = False,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:

        super().__init__()

        self.datasets = datasets

        self.fold = fold
        self.invert_fold_selection = invert_fold_selection

        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sampler = sampler

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
        dataset = instantiate(
            self.datasets[mode],
            mode=mode,
            fold=fold,
            invert_fold_selection=self.invert_fold_selection,
        )

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

        if hasattr(self.train, "get_labels") and callable(self.train.get_labels):
            return self.train.get_labels()

        raise RuntimeError("Train dataset does not provide labels.")

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
