from collections.abc import Iterable, Sequence
from typing import overload

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from ml.datamodule.datasets.base import (
    LabeledBagOfTilesDataset,
    UnlabeledBagOfTilesDataset,
)
from ml.typing import (
    Bag,
    BagBatch,
    BagMetadata,
    FullyLabeledBag,
    FullyLabeledBagBatch,
    UnlabeledBag,
    UnlabeledBagBatch,
    WeaklyLabeledBag,
    WeaklyLabeledBagBatch,
)


@overload
def collate_fn(batch: list[FullyLabeledBag]) -> FullyLabeledBagBatch: ...


@overload
def collate_fn(batch: list[WeaklyLabeledBag]) -> WeaklyLabeledBagBatch: ...


@overload
def collate_fn(batch: list[UnlabeledBag]) -> UnlabeledBagBatch: ...


def collate_fn(batch: Sequence[Bag]) -> BagBatch:

    transposed_batch = list(zip(*batch, strict=True))

    features: list[torch.Tensor] = list(transposed_batch[0])
    metadata: list[BagMetadata] = list(transposed_batch[1])
    labels: list[tuple[torch.Tensor]] = transposed_batch[2:]

    stacked_labels = (
        torch.stack(l)
        if l[0].ndim == 0
        else pad_sequence(list(l), batch_first=True, padding_value=-1)
        for l in labels
    )

    lengths = torch.tensor([f.shape[0] for f in features])
    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)
    padding_mask = torch.arange(max(lengths)).unsqueeze(0) < lengths.unsqueeze(1)

    return padded_features, padding_mask, list(metadata), *stacked_labels


class BagsDataModule[B, BB](LightningDataModule):
    train: LabeledBagOfTilesDataset[B]
    val: LabeledBagOfTilesDataset[B]
    test: LabeledBagOfTilesDataset[B]
    predict: UnlabeledBagOfTilesDataset

    def __init__(
        self,
        batch_size: int,
        shuffle: bool,
        sampler: DictConfig | None = None,
        num_workers: int = 0,
        **datasets: DictConfig,
    ) -> None:

        super().__init__()

        self.datasets = datasets
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

    def train_dataloader(self) -> Iterable[BB]:
        sampler = (
            instantiate(self.sampler, labels=self.train.get_labels())
            if self.sampler is not None
            else None
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            sampler=sampler,
            shuffle=sampler is None and self.shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[BB]:
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Iterable[BB]:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> Iterable[UnlabeledBagBatch]:
        return DataLoader(
            self.predict,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=self.num_workers,
        )


class WeaklyLabeledBagsDataModule(
    BagsDataModule[WeaklyLabeledBag, WeaklyLabeledBagBatch]
): ...


class FullyLabeledBagsDataModule(
    BagsDataModule[FullyLabeledBag, FullyLabeledBagBatch]
): ...
