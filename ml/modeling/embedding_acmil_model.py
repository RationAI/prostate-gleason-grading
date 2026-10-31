from typing import Any, cast, override

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from ml.modeling.base import SLGleasonModel
from ml.typing import UnlabeledBagBatch, WeaklyLabeledBagBatch


class MLP(nn.Module):
    def __init__(self, dims: list[int]) -> None:

        super().__init__()

        layers: list[nn.Module] = []

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class STKIMAttentionBranch(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_dim: int,
        mlp_dims: list[int],
        stkim_k: int,
        stkim_p: float,
    ) -> None:

        super().__init__()

        self.stkim_k = stkim_k
        self.stkim_p = stkim_p

        self.v = nn.Sequential(nn.Linear(embedding_dim, attention_dim), nn.Tanh())
        self.u = nn.Sequential(nn.Linear(embedding_dim, attention_dim), nn.Sigmoid())
        self.w = nn.Linear(attention_dim, 1, bias=False)

        self.mlp = MLP([embedding_dim, *mlp_dims, 1])

    def attention(self, x: Tensor, mask: Tensor) -> Tensor:

        # B ~ batch size, N ~ number of tiles, D ~ embeddings dim
        # A ~ attention dim, C ~ number of classes

        # x has shape [B, N, D], mask has shape [B, N] and is boolean

        v = self.v(x)  # [B, N, A]
        u = self.u(x)  # [B, N, A]
        w = self.w(v * u).squeeze(-1)  # [B, N]
        w = w.masked_fill(~mask, float("-inf"))  # [B, N], -inf for padding tiles
        a = torch.softmax(w, dim=1)  # [B, N], 0 for padding tiles

        if self.training and self.stkim_k > 0 and self.stkim_p > 0:
            k, p = min(self.stkim_k, a.shape[1]), self.stkim_p

            top_k_indices = torch.topk(a, k=k, dim=1).indices

            drop_mask = torch.rand(a.shape[0], k, device=a.device) < p
            drop_mask = torch.zeros_like(a, dtype=torch.bool).scatter(
                dim=1, index=top_k_indices, src=drop_mask
            )

            a = a.masked_fill(drop_mask, 0.0)
            a = a / (a.sum(dim=1, keepdim=True))

        return a

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        attention = self.attention(x, mask)  # [B, N]
        embedding = (x * attention.unsqueeze(-1)).sum(dim=1)  # [B, D]
        prediction = torch.sigmoid(self.mlp(embedding).squeeze(-1))  # [B]
        return prediction, embedding, attention


class EmbeddingAttentionChallengingMILGleasonModel(SLGleasonModel):
    def __init__(
        self,
        lr: float,
        adamw_kwargs: dict[str, Any],
        num_classes: int,
        embedding_dim: int,
        mlp_hidden_dim: list[int],
        attention_dim: int = 512,
        stkim_k: int = 10,
        stkim_p: float = 0.6,
        loss_p_weight: float = 1.0,
        loss_d_weight: float = 1.0,
        loss_b_weight: float = 1.0,
        weight_decay: float = 0.0,
    ) -> None:

        super().__init__(num_classes)

        self.lr = lr
        self.weight_decay = weight_decay
        self.adamw_kwargs = adamw_kwargs

        self.loss_p_weight = loss_p_weight
        self.loss_d_weight = loss_d_weight
        self.loss_b_weight = loss_b_weight

        self.mlp = MLP([embedding_dim, *mlp_hidden_dim, num_classes])

        self.attention_branches = nn.ModuleList(
            [
                STKIMAttentionBranch(
                    embedding_dim,
                    attention_dim,
                    mlp_hidden_dim,
                    stkim_k,
                    stkim_p,
                )
                for _ in range(4)  # one per gleason pattern (negative, 3, 4, 5)
            ]
        )

    def forward(
        self, x: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:

        branch_predictions: list[Tensor] = []
        branch_attentions: list[Tensor] = []

        slide_embedding = x.new_zeros(x.shape[0], x.shape[-1])  # [B, D]
        slide_attention = x.new_zeros(x.shape[0], x.shape[1])  # [B, N]

        for branch in self.attention_branches:
            branch_prediction, branch_embedding, branch_attention = branch(x, mask)

            branch_predictions.append(branch_prediction)
            branch_attentions.append(branch_attention)
            slide_embedding += branch_embedding
            slide_attention += branch_attention

        slide_embedding /= len(self.attention_branches)
        slide_attention /= len(self.attention_branches)

        slide_logits = self.mlp(slide_embedding)  # [B, C]

        # when visualizing, multiplying the branch prediction with the branch
        # attention heatmap can help to interpret heatmaps as evidence for each pattern

        return slide_logits, slide_attention, branch_predictions, branch_attentions

    def gleason_score_to_gleason_patterns(self, labels: Tensor) -> Tensor:

        labels_map = cast("Any", self.trainer).datamodule.train.labels_map_inverse
        if labels_map is None:
            raise RuntimeError("Labels are expected to be invertible.")

        gleason_patterns = torch.zeros(
            (len(labels), 3), device=labels.device, dtype=torch.float32
        )

        for i, label in enumerate(labels):
            gleason_score = labels_map[label.item()]
            for gleason_pattern in ["3", "4", "5"]:
                if gleason_pattern in gleason_score:
                    gleason_patterns[i, int(gleason_pattern) - 3] = 1

        return gleason_patterns

    def loss_b(self, logits: Tensor, targets: Tensor) -> Tensor:
        # SL Gleason score classification loss
        return self.criterion(logits, targets)

    def loss_p(self, branch_preds: list[Tensor], targets: Tensor) -> Tensor:

        # loss encouraging pattern-specific branch predictions

        targets = self.gleason_score_to_gleason_patterns(targets)

        loss = torch.tensor(0, device=targets.device, dtype=torch.float32)

        for i, preds in enumerate(branch_preds[1:]):  # ignore negative branch
            labels = targets[:, i]
            loss += F.binary_cross_entropy(preds, labels)

        return loss / (len(branch_preds) - 1)

    def loss_d(self, branch_attentions: list[Tensor]) -> Tensor:

        # loss encouraging non-overlapping branch attentions

        loss = torch.tensor(0, device=branch_attentions[0].device, dtype=torch.float32)

        for i, attention_i in enumerate(branch_attentions):
            for attention_j in branch_attentions[i + 1 :]:
                loss += F.cosine_similarity(attention_i, attention_j, dim=-1).mean()

        branches_count = len(branch_attentions)

        return loss * 2 / (branches_count * (branches_count - 1))

    def loss(
        self,
        stage: str,
        slide_logits: Tensor,
        slide_targets: Tensor,
        outputs: dict[str, Any],
    ) -> Tensor:

        log_kwargs: dict[str, Any] = {
            "batch_size": slide_logits.shape[0],
            "on_step": stage == "train",
            "on_epoch": True,
            "prog_bar": True,
        }

        loss_b = self.loss_b(slide_logits, slide_targets)
        self.log(f"{stage}/loss_b", loss_b, **log_kwargs)

        loss_p = self.loss_p(outputs["branch_predictions"], slide_targets)
        self.log(f"{stage}/loss_p", loss_p, **log_kwargs)

        loss_d = self.loss_d(outputs["branch_attentions"])
        self.log(f"{stage}/loss_d", loss_d, **log_kwargs)

        loss = (
            loss_b * self.loss_b_weight
            + loss_p * self.loss_p_weight
            + loss_d * self.loss_d_weight
        )

        self.log(f"{stage}/loss", loss, **log_kwargs)

        return loss

    @override
    def training_step(
        self, batch: WeaklyLabeledBagBatch, batch_idx: int
    ) -> Tensor | None:
        logits, targets, outputs = self.process_train_batch(batch)
        self.train_cm.update(logits, targets)
        self.train_metrics.update(logits, targets)
        return self.loss("train", logits, targets, outputs)

    @override
    def validation_step(self, batch: WeaklyLabeledBagBatch) -> None:
        logits, targets, outputs = self.process_train_batch(batch)
        self.val_cm.update(logits, targets)
        self.val_metrics.update(logits, targets)
        self.loss("validation", logits, targets, outputs)

    def process_train_batch(
        self, batch: WeaklyLabeledBagBatch
    ) -> tuple[Tensor, Tensor, dict[str, Any]]:
        embeddings, padding_mask, metadata, targets = batch
        logits, attention, branch_preds, branch_attention = self(
            embeddings, padding_mask
        )
        return (
            logits,
            targets,
            {
                "metadata": metadata,
                "slide_attention": attention,
                "branch_predictions": branch_preds,
                "branch_attentions": branch_attention,
            },
        )

    def process_predict_batch(
        self, batch: UnlabeledBagBatch
    ) -> tuple[Tensor, dict[str, Any]]:
        embeddings, padding_mask, metadata = batch
        logits, attention, branch_preds, branch_attention = self(
            embeddings, padding_mask
        )
        return (
            logits,
            {
                "metadata": metadata,
                "slide_attention": attention,
                "branch_predictions": branch_preds,
                "branch_attentions": branch_attention,
            },
        )

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.adamw_kwargs,
        )
