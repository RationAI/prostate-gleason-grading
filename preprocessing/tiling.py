from typing import cast

import hydra
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger


@hydra.main(config_path="./configs", config_name="preproessing/tiling", version_base=None)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)

    logger.log_artifacts(logger.run_id, config.tiling.src_path, config.dest_path)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
