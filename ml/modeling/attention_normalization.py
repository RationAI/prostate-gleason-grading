from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AttentionNormalization(ABC, nn.Module):
    @abstractmethod
    def forward(self, a: Tensor, mask: Tensor) -> Tensor:
        pass


class Softmax(AttentionNormalization):
    def forward(self, a: Tensor, mask: Tensor) -> Tensor:
        a = a.masked_fill(~mask, float("-inf"))
        return torch.softmax(a, dim=1)


class SigmoidSoftmax(AttentionNormalization):
    def forward(self, a: Tensor, mask: Tensor) -> Tensor:
        a = torch.sigmoid(a)
        a = a.masked_fill(~mask, float("-inf"))
        return torch.softmax(a, dim=1)


class SoftplusNormalization(AttentionNormalization):
    def forward(self, a: Tensor, mask: Tensor) -> Tensor:
        sp = F.softplus(a)
        sp = sp.masked_fill(~mask, 0.0)
        return sp / sp.sum(dim=1, keepdim=True)
