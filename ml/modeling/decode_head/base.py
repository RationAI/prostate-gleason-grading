from abc import ABC, abstractmethod

from torch import Tensor, nn


class Classifier(ABC, nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_probability: float,
    ) -> None:

        super().__init__()

        self.dropout = nn.Dropout(p=dropout_probability)
        self.proj = nn.Linear(in_features, out_features)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass
