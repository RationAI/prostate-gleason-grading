import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


@with_cli_args(["+preprocessing=add_gleason_score"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slide_metadata = pd.read_csv(
        mlflow.artifacts.download_artifacts(config.dataset.mlflow_uris.slide_metadata)
    )

    tiling_uri = config.dataset.mlflow_uris.tiling
    slides_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/slides.parquet")
    )
    tiles_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/tiles.parquet")
    )

    slides_df = slides_df.join(
        slide_metadata.set_index("slide_path")["gleason_score"],
        on="path",
    )

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.dataset.name,
    )


if __name__ == "__main__":
    main()
