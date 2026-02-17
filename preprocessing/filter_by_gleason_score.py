import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


@with_cli_args(["+preprocessing=filter_by_gleason_score"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tiling_uri = config.dataset.mlflow_uris.tiling_with_gs
    slides_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/slides.parquet")
    )
    tiles_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/tiles.parquet")
    )

    gleason_scores_to_keep = set(config.gleason_scores_to_keep)
    slides_df = slides_df[slides_df["gleason_score"].isin(gleason_scores_to_keep)]
    tiles_df = tiles_df[tiles_df["slide_id"].isin(slides_df["id"])]

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.dataset.name,
    )


if __name__ == "__main__":
    main()
