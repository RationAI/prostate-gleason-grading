from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from ml.modeling.base import SLGleasonModel
from ml.modeling.decode_head import Classifier
from ml.typing import UnlabeledBagBatch, WeaklyLabeledBagBatch


class EmbeddingAttentionBasedMILGleasonModel(SLGleasonModel):
    def __init__(
        self,
        num_classes: int,
        classifier: Classifier,
        lr: float,
        adamw_kwargs: dict[str, Any],
        attention_dim: int = 512,
        weight_decay: float = 0.0,
    ) -> None:

        super().__init__(num_classes)

        self.classifier = classifier

        self.lr = lr
        self.weight_decay = weight_decay
        self.adamw_kwargs = adamw_kwargs

        in_dim = classifier.in_features
        self.attention_V = nn.Sequential(nn.Linear(in_dim, attention_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(in_dim, attention_dim), nn.Sigmoid())
        self.attention_W = nn.Linear(attention_dim, 1, bias=False)

    def attention(self, x: Tensor, mask: Tensor) -> Tensor:

        # B ~ batch size, N ~ number of tiles, D ~ embeddings dim
        # A ~ attention dim, C ~ number of classes

        # x has shape [B, N, D], mask has shape [B, N] and is boolean

        v = self.attention_V(x)  # [B, N, A]
        u = self.attention_U(x)  # [B, N, A]
        w = self.attention_W(v * u).squeeze(-1)  # [B, N]
        w = w.masked_fill(~mask, float("-inf"))  # [B, N], -inf for padding tiles
        return torch.softmax(w, dim=1)  # [B, N], 0 for padding tiles

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        attention = self.attention(x, mask)  # [B, N]
        tile_logits = self.classifier(x)  # [B, N, C]
        slide_logits = (tile_logits * attention.unsqueeze(-1)).sum(dim=1)  # [B, C]
        return slide_logits, tile_logits, attention

    def process_train_batch(
        self, batch: WeaklyLabeledBagBatch
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        embeddings, padding_mask, metadata, targets = batch
        slide_logits, tile_logits, attention = self(embeddings, padding_mask)
        return (
            slide_logits,
            targets,
            {"metadata": metadata, "tile_logits": tile_logits, "attention": attention},
        )

    def process_predict_batch(
        self, batch: UnlabeledBagBatch
    ) -> tuple[Tensor, dict[str, Any]]:
        embeddings, padding_mask, metadata = batch
        slide_logits, tile_logits, attention = self(embeddings, padding_mask)
        return (
            slide_logits,
            {"metadata": metadata, "tile_logits": tile_logits, "attention": attention},
        )

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.adamw_kwargs,
        )
