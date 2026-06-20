import torch
from lightning import Callback, LightningModule, Trainer


class GradientNormStopping(Callback):
    def __init__(
        self,
        tolerance: float = 1e-8,
        norm_type: int | float = float("inf"),
    ) -> None:
        self.tolerance = tolerance
        self.norm_type = norm_type

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:

        gradient = [
            p.grad.flatten() for p in pl_module.parameters() if p.grad is not None
        ]

        if not gradient:
            return

        gradient_norm = torch.linalg.vector_norm(
            torch.cat(gradient),
            ord=self.norm_type,
        )

        pl_module.log(
            "gradient_norm",
            gradient_norm.detach(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if gradient_norm.item() < self.tolerance:
            trainer.should_stop = True
