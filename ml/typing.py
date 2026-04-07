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


type LabeledSample = tuple[Tensor, Metadata, str]
type UnlabeledSample = tuple[Tensor, Metadata]

type LabeledSampleBatch = tuple[Tensor, Tensor, MetadataBatch]
type UnlabeledSampleBatch = tuple[Tensor, MetadataBatch]

type Input = Tensor

type Outputs = Tensor
