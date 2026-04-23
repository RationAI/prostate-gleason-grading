from collections.abc import Iterable
from typing import cast

from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from rationai.mlkit.data.datasets import MetaTiledSlides
from torch.utils.data import DataLoader

from ml.typing import LabeledSample, UnlabeledSample


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        validation_fold: int | None = None,
        sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = validation_fold
        self.sampler = sampler
        self.datasets = datasets

    def _instantiate_sampler(
        self, dataset: MetaTiledSlides[LabeledSample]
    ) -> Iterable[int] | None:
        return (
            instantiate(self.sampler, slides_dataset=dataset)
            if self.sampler is not None
            else None
        )

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

    def train_dataloader(self) -> Iterable[LabeledSample]:
        sampler = self._instantiate_sampler(self.train)
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=True if sampler is None else None,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[LabeledSample]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> list[Iterable[LabeledSample]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.test.datasets
        ]

    def predict_dataloader(self) -> list[Iterable[UnlabeledSample]]:
        return [
            DataLoader(
                dataset, batch_size=self.batch_size, num_workers=self.num_workers
            )
            for dataset in self.predict.datasets
        ]
