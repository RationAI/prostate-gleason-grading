from torch import Tensor, bincount
from torch.utils.data import WeightedRandomSampler


class StratifiedWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        labels: Tensor,
        num_samples: int | None = None,
        replacement: bool = True,
    ) -> None:

        num_total_samples = labels.numel()
        num_class_samples = bincount(labels).double().clamp_min(1.0)

        if (
            not replacement
            and num_samples is not None
            and num_samples > num_total_samples
        ):
            raise ValueError(
                "The number of samples can't exceed the size of the "
                "dataset when samples are drawn without replacement."
            )

        class_weights = num_total_samples / num_class_samples
        sample_weights = class_weights[labels]

        super().__init__(
            weights=sample_weights.tolist(),
            num_samples=num_samples or num_total_samples,
            replacement=replacement,
        )
