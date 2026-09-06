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


type LabeledSample = tuple[Tensor, Metadata, Tensor]
type UnlabeledSample = tuple[Tensor, Metadata]

type LabeledSampleBatch = tuple[Tensor, MetadataBatch, Tensor]
type UnlabeledSampleBatch = tuple[Tensor, MetadataBatch]

type FullyLabeledBag = tuple[Tensor, BagMetadata, Tensor, Tensor]
type WeaklyLabeledBag = tuple[Tensor, BagMetadata, Tensor]
type UnlabeledBag = tuple[Tensor, BagMetadata]
