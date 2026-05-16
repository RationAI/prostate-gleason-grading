from torch import Tensor, nn

from ml.base import GleasonModel


class EmbeddingGleasonModel(GleasonModel):
    def __init__(
        self,
        num_classes: int,
        lr: float,
        decode_head: nn.Module,
    ) -> None:

        super().__init__(num_classes, lr)
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        return self.decode_head(x)
