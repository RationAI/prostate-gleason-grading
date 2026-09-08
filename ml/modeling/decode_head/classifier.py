from torch import Tensor, nn


class Classifier(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout_probability: float,
    ) -> None:

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout_probability = dropout_probability

        self.dropout = nn.Dropout(p=dropout_probability)
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.dropout(x))
