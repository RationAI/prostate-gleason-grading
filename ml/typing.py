from typing import TypedDict

from torch import Tensor


class Metadata(TypedDict):
    slide: str
    x: int
    y: int


class MetadataBatch(TypedDict):
    slide: list[str]
    x: Tensor
    y: Tensor


class BagMetadata(TypedDict):
    slide: str
    x: Tensor
    y: Tensor


type UnlabeledSample = tuple[Tensor, Metadata]
type LabeledSample = tuple[Tensor, Metadata, Tensor]

type UnlabeledSampleBatch = tuple[Tensor, MetadataBatch]
type LabeledSampleBatch = tuple[Tensor, MetadataBatch, Tensor]

type Bag = tuple[Tensor, BagMetadata, *tuple[Tensor, ...]]
type UnlabeledBag = tuple[Tensor, BagMetadata]
type WeaklyLabeledBag = tuple[Tensor, BagMetadata, Tensor]
type FullyLabeledBag = tuple[Tensor, BagMetadata, Tensor, Tensor]

type BagBatch = tuple[Tensor, Tensor, list[BagMetadata], *tuple[Tensor, ...]]
type UnlabeledBagBatch = tuple[Tensor, Tensor, list[BagMetadata]]
type WeaklyLabeledBagBatch = tuple[Tensor, Tensor, list[BagMetadata], Tensor]
type FullyLabeledBagBatch = tuple[Tensor, Tensor, list[BagMetadata], Tensor, Tensor]
