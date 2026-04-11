from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import mlflow
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
        **datasets: DictConfig,
    ) -> None:

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = validation_fold
        self.datasets = datasets

    def _instantiate_dataset(
        self, mode: str, **dataset_kwargs: Any
    ) -> MetaTiledSlides[LabeledSample] | MetaTiledSlides[UnlabeledSample] | None:

        if mode == "val" and self.fold is None:
            return None

        if mode in {"train", "val"}:
            dataset = instantiate(
                self.datasets[mode], fold=self.fold, mode=mode, **dataset_kwargs
            )
        else:
            dataset = instantiate(self.datasets[mode], **dataset_kwargs)

        return (
            cast("MetaTiledSlides[LabeledSample]", dataset)
            if mode == "test"
            else cast("MetaTiledSlides[UnlabeledSample]", dataset)
        )

    def setup(self, stage: str, **dataset_kwargs: Any) -> None:
        match stage:
            case "fit":
                self.train = self._instantiate_dataset("train", **dataset_kwargs)
                self.val = self._instantiate_dataset("val", **dataset_kwargs)
            case "validate":
                self.val = self._instantiate_dataset("val", **dataset_kwargs)
            case "test":
                self.test = self._instantiate_dataset("test", **dataset_kwargs)
            case "predict":
                self.predict = self._instantiate_dataset("predict", **dataset_kwargs)

    def train_dataloader(self) -> Iterable[LabeledSample]:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[LabeledSample] | None:
        return (
            None
            if self.val is None
            else DataLoader(
                self.val,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                persistent_workers=self.num_workers > 0,
            )
        )

    def test_dataloader(self) -> Iterable[LabeledSample]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> Iterable[UnlabeledSample]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class EmbeddingsDataModule(DataModule):
    def __init__(
        self,
        batch_size: int,
        embeddings_uri: str,
        embeddings_dir: str,
        num_workers: int = 0,
        validation_fold: int | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__(batch_size, num_workers, validation_fold, **datasets)
        self.embeddings_uri = embeddings_uri
        self.embeddings_dir = Path(embeddings_dir)

    def prepare_data(self) -> None:
        mlflow.artifacts.download_artifacts(
            self.embeddings_uri,
            dst_path=str(self.embeddings_dir),
        )

    def setup(self, stage: str, **dataset_kwargs: Any) -> None:
        super().setup(
            stage,
            embeddings_dir=self.embeddings_dir / Path(self.embeddings_uri).stem,
            **dataset_kwargs,
        )
