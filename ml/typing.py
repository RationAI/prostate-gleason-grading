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


type LabeledSample = tuple[Tensor, Metadata, Tensor]
type UnlabeledSample = tuple[Tensor, Metadata]

type LabeledSampleBatch = tuple[Tensor, MetadataBatch, Tensor]
type UnlabeledSampleBatch = tuple[Tensor, MetadataBatch]
