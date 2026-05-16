from torch import Tensor

from ml.modeling.decode_head.base import Classifier


class EmbeddingClassifier(Classifier):
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {x.ndim}D")
        return self.proj(self.dropout(x))
