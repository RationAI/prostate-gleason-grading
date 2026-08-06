from collections.abc import Iterable
from typing import Literal, cast, overload

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
    ) -> LabeledSlideDataset: ...

    @overload
    def _instantiate_dataset(
        self, mode: Literal["predict"]
    ) -> UnlabeledSlideDataset: ...

    def _instantiate_dataset(
        self, mode: str
    ) -> LabeledSlideDataset | UnlabeledSlideDataset:

        fold = self.fold if mode in {"train", "val"} else None
        invert = self.invert_fold_selection ^ (mode != "val")

        dataset = instantiate(
            self.datasets[mode],
            fold=fold,
            invert_fold_selection=invert,
        )

        return (
            cast("UnlabeledSlideDataset", dataset)
            if mode == "predict"
            else cast("LabeledSlideDataset", dataset)
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
