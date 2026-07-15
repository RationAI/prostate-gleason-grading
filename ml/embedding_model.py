from typing import TYPE_CHECKING, Any, cast, override

import torch
from torch import Tensor, nn
from torch.optim import LBFGS, Optimizer

from ml.base import GleasonModel
from ml.modeling.decode_head.base import Classifier
from ml.typing import LabeledSampleBatch


if TYPE_CHECKING:
    from ml.datamodule.data_module import DataModule


class EmbeddingGleasonModel(GleasonModel):
    def __init__(
        self,
        num_classes: int,
        lr: float,
        decode_head: Classifier,
    ) -> None:

        super().__init__(num_classes, lr)
        self.decode_head = decode_head

    def forward(self, x: Tensor) -> Tensor:
        return self.decode_head(x)


class LBFGSEmbeddingsGleasonModel(EmbeddingGleasonModel):
    def __init__(
        self,
        num_classes: int,
        lr: float,
        decode_head: Classifier,
        lbfgs_kwargs: dict[str, Any],
        weight_decay: float = 0.0,
        cache_on_cpu: bool = True,
        early_stopping: bool = True,
    ) -> None:

        super().__init__(num_classes, lr, decode_head)

        self.early_stopping = early_stopping

        self.automatic_optimization = False

        self._weight_decay = weight_decay
        self._lbfgs_kwargs = lbfgs_kwargs
        self._cache_on_cpu = cache_on_cpu
        self._batch_cache: list[tuple[Tensor, Tensor]] = []

    @override
    def configure_optimizers(self) -> Optimizer:
        return LBFGS(
            self.parameters(),
            lr=self.lr,
            line_search_fn="strong_wolfe",
            **self._lbfgs_kwargs,
        )

    def _configure_criterion(self) -> None:

        labels = cast("Any", self.trainer).datamodule.get_train_labels()

        num_total_samples = labels.numel()
        num_class_samples = (
            torch.bincount(
                labels,
                minlength=self.num_classes,
            )
            .float()
            .clamp_min(1.0)
        )

        class_frequencies = num_class_samples / num_total_samples
        class_weights = 1 / (self.num_classes * class_frequencies)

        self.criterion = nn.CrossEntropyLoss(
            reduction="mean",
            weight=class_weights.to(self.device),
        )

    def _validate_requirements(self) -> None:

        classifier: Classifier = self.decode_head
        datamodule: DataModule = cast("Any", self.trainer).datamodule

        samples_per_epoch = self.trainer.num_training_batches * datamodule.batch_size

        if samples_per_epoch < len(datamodule.train) or datamodule.drop_last:
            raise ValueError(
                "LBFGS requires global deterministic objective function. "
                "Dropping last batch or not covering the whole dataset is "
                "not allowed."
            )

        if (
            classifier.dropout_probability > 0
            or datamodule.shuffle
            or datamodule.sampler is not None
        ):
            raise ValueError(
                "LBFGS requires global deterministic objective function. "
                "Any non-determinism like shuffling, randomized sampling, "
                "or classifier dropout is not allowed."
            )

    def _compute_loss_and_backward(self, dataset_weight: Tensor) -> Tensor:

        dataset_loss = torch.zeros((), device=self.device)

        for x, y in self._batch_cache:
            if self._cache_on_cpu:
                x = x.to(self.device)
                y = y.to(self.device)

            assert self.criterion.weight is not None
            batch_weight = self.criterion.weight[y].sum()
            batch_loss = self.criterion(self(x), y)

            rescaled_loss = batch_loss * batch_weight / dataset_weight

            if not torch.isfinite(rescaled_loss):
                raise FloatingPointError("Non-finite loss.")

            self.manual_backward(rescaled_loss, retain_graph=False)

            dataset_loss += rescaled_loss.detach()

        l2_penalty = self.decode_head.proj.weight.square().sum()
        regularization_loss = 1 / 2 * self._weight_decay * l2_penalty

        self.manual_backward(regularization_loss, retain_graph=False)

        return (dataset_loss + regularization_loss).detach()

    def _update_and_log_metrics(self, num_samples: int) -> None:

        self.train_metrics.reset()
        with torch.no_grad():
            for x, y in self._batch_cache:
                if self._cache_on_cpu:
                    x = x.to(self.device)
                    y = y.to(self.device)
                self.train_metrics.update(self(x), y)

            self.log_dict(self.train_metrics, batch_size=num_samples, on_epoch=True)

    def _get_n_iters(self, optimizer: LBFGS) -> int:
        param = optimizer.param_groups[0]["params"][0]
        return optimizer.state[param].get("n_iter", 0)

    def on_fit_start(self) -> None:
        self._configure_criterion()

    def on_train_start(self) -> None:
        self._validate_requirements()

    @override
    def training_step(self, batch: LabeledSampleBatch) -> None:

        x, _, y = batch

        if self._cache_on_cpu:
            x = x.cpu()
            y = y.cpu()

        self._batch_cache.append((x, y))

    def on_train_epoch_start(self) -> None:
        self._batch_cache.clear()

    def on_train_epoch_end(self) -> None:

        optimizer = cast("LBFGS", self.optimizers())

        num_samples = sum(len(y) for _, y in self._batch_cache)
        assert num_samples == len(cast("Any", self.trainer).datamodule.train)

        assert self.criterion.weight is not None
        dataset_weight = torch.zeros((), device=self.device)
        for _, y in self._batch_cache:
            idx = y.to(self.device) if self._cache_on_cpu else y
            dataset_weight += self.criterion.weight[idx].sum()

        def closure() -> Tensor:
            optimizer.zero_grad()
            return self._compute_loss_and_backward(dataset_weight)

        prev_n_iters = self._get_n_iters(optimizer)

        loss = optimizer.step(closure)

        curr_n_iters = self._get_n_iters(optimizer)
        step_n_iters = curr_n_iters - prev_n_iters

        self.log("train/loss", loss, batch_size=num_samples, on_epoch=True)
        self._update_and_log_metrics(num_samples)

        if self.early_stopping and step_n_iters < optimizer.param_groups[0]["max_iter"]:
            self.trainer.should_stop = True
