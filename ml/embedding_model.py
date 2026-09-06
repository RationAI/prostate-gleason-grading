from typing import TYPE_CHECKING, Any, cast, override

import lightning as pl
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

        self.automatic_optimization = False

        self.early_stopping = early_stopping

        self._weight_decay = weight_decay
        self._lbfgs_kwargs = lbfgs_kwargs
        self._cache_on_cpu = cache_on_cpu
        self._batch_cache: list[tuple[Tensor, Tensor]] = []

        # "dummy" criterion for checkpoint loading, weights will be
        # recomputed based on the label distribution during training

        self._train_criterion = nn.CrossEntropyLoss(
            reduction="mean",
            weight=torch.ones(num_classes),
        )

    @override
    def configure_optimizers(self) -> Optimizer:
        return LBFGS(
            self.parameters(),
            lr=self.lr,
            line_search_fn="strong_wolfe",
            **self._lbfgs_kwargs,
        )

    def _configure_train_criterion(self) -> None:

        labels = cast("Any", self.trainer).datamodule.train.get_tile_labels()

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

        self._train_criterion = nn.CrossEntropyLoss(
            reduction="mean",
            weight=class_weights.to(self.device),
        )

    def _validate_requirements(self) -> None:

        trainer: pl.Trainer = self.trainer
        datamodule: DataModule = cast("Any", trainer).datamodule
        classifier: Classifier = self.decode_head

        samples_per_epoch = trainer.num_training_batches * datamodule.batch_size

        if (
            samples_per_epoch < len(datamodule.train)
            or datamodule.drop_last
            or trainer.limit_train_batches != 1.0
        ):
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

        if trainer.num_devices != 1 or trainer.num_nodes != 1:
            raise ValueError("LBFGS requires a single device on a single node.")

        if trainer.val_check_interval != 1.0:
            print(
                "Warning: Training is performed only once at the end of the "
                "epoch. Validation between training steps is not meaningful."
            )

    def _compute_loss_and_backward(
        self, dataset_weight: Tensor, batch_weights: list[Tensor]
    ) -> Tensor:

        dataset_loss = torch.zeros((), device=self.device)

        for (x, y), batch_weight in zip(self._batch_cache, batch_weights, strict=True):
            if self._cache_on_cpu:
                x = x.to(self.device)
                y = y.to(self.device)

            batch_loss = self._train_criterion(self(x), y)

            rescaled_loss = batch_loss * batch_weight / dataset_weight

            if not torch.isfinite(rescaled_loss):
                raise FloatingPointError("Non-finite loss.")

            self.manual_backward(rescaled_loss, retain_graph=False)

            dataset_loss += rescaled_loss.detach()

        l2_penalty = self.decode_head.proj.weight.square().sum()
        regularization_loss = 1 / 2 * self._weight_decay * l2_penalty

        self.manual_backward(regularization_loss, retain_graph=False)

        return (dataset_loss + regularization_loss).detach()

    def _update_metrics(self) -> None:

        with torch.no_grad():
            for x, y in self._batch_cache:
                if self._cache_on_cpu:
                    x = x.to(self.device)
                    y = y.to(self.device)

                logits = self(x)

                self.train_cm.update(logits, y)
                self.train_metrics.update(logits, y)

    def _get_n_iters(self, optimizer: LBFGS) -> int:
        param = optimizer.param_groups[0]["params"][0]
        return optimizer.state[param].get("n_iter", 0)

    def _optimize(self) -> Tensor:

        optimizer = cast("LBFGS", self.optimizers())

        assert self._train_criterion.weight is not None
        batch_weights: list[Tensor] = []
        for _, y in self._batch_cache:
            idx = y.to(self.device) if self._cache_on_cpu else y
            batch_weights.append(self._train_criterion.weight[idx].sum())
        dataset_weight = sum(batch_weights, start=torch.zeros((), device=self.device))

        def closure() -> Tensor:
            optimizer.zero_grad()
            return self._compute_loss_and_backward(dataset_weight, batch_weights)

        prev_n_iters = self._get_n_iters(optimizer)

        loss = optimizer.step(closure)

        curr_n_iters = self._get_n_iters(optimizer)
        step_n_iters = curr_n_iters - prev_n_iters

        if self.early_stopping and step_n_iters < optimizer.param_groups[0]["max_iter"]:
            self.trainer.should_stop = True

        return loss

    def on_fit_start(self) -> None:
        self._configure_train_criterion()

    def on_train_start(self) -> None:
        self._validate_requirements()

    def on_train_epoch_start(self) -> None:
        self._batch_cache.clear()

    @override
    def training_step(self, batch: LabeledSampleBatch, batch_idx: int) -> None:

        x, _, y = batch

        if self._cache_on_cpu:
            x = x.cpu()
            y = y.cpu()

        self._batch_cache.append((x, y))

        if batch_idx == self.trainer.num_training_batches - 1:
            num_samples = sum(len(y) for _, y in self._batch_cache)
            assert num_samples == len(cast("Any", self.trainer).datamodule.train)

            loss = self._optimize()
            self.log(
                "train/loss",
                loss,
                batch_size=num_samples,
                on_step=False,
                on_epoch=True,
            )

            self._update_metrics()

            self._batch_cache.clear()
