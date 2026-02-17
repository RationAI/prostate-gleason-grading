import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


@with_cli_args(["+preprocessing=merge_tiling_sources"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    slides_dfs, tiles_dfs = [], []

    for tiling_source_uri in config.dataset.mlflow_uris.tiling_sources:
        slides_dfs.append(
            pd.read_parquet(
                mlflow.artifacts.download_artifacts(
                    tiling_source_uri + "/slides.parquet"
                )
            )
        )
        tiles_dfs.append(
            pd.read_parquet(
                mlflow.artifacts.download_artifacts(
                    tiling_source_uri + "/tiles.parquet"
                )
            )
        )

    slides_df = pd.concat(slides_dfs, ignore_index=True)

    if not slides_df["id"].is_unique:
        raise ValueError("Slide IDs are expected not to overlap.")

    tiles_df = pd.concat(tiles_dfs, ignore_index=True)

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.dataset.name,
    )


if __name__ == "__main__":
    main()
