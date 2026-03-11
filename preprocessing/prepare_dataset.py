import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset


@with_cli_args(["+preprocessing=prepare_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    annotations = pd.read_json(
        mlflow.artifacts.download_artifacts(
            config.dataset.mlflow_uris.cancer_prediction_table
        ),
        orient="split",
    )

    tiling_uri = config.dataset.mlflow_uris.tiling_with_gs
    slides_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/slides.parquet")
    )
    tiles_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/tiles.parquet")
    )

    negative_slides_df = slides_df[
        slides_df.gleason_score.isna() & ~slides_df.carcinoma
    ]
    positive_slides_df = slides_df[
        slides_df.gleason_score.isin(config.gleason_scores_to_keep)
        & slides_df.carcinoma
    ]

    tiles_df = tiles_df[
        (tiles_df.tissue_roi_percentage > config.dataset.thresholds.tissue)
        & (tiles_df.blur_percentage <= config.dataset.thresholds.blur)
        & (tiles_df.folding_percentage <= config.dataset.thresholds.folding)
        & (tiles_df.residual_percentage <= config.dataset.thresholds.residual)
    ]

    negative_tiles_df = tiles_df[tiles_df.slide_id.isin(negative_slides_df.id)].copy()
    positive_tiles_df = tiles_df[tiles_df.slide_id.isin(positive_slides_df.id)].copy()

    annotations = annotations.join(slides_df.set_index("stem")["id"], on="slide")

    positive_tiles_df = positive_tiles_df.join(
        annotations.set_index(["id", "x", "y"])["binary_prediction"],
        on=["slide_id", "x", "y"],
        how="inner",
    )
    negative_tiles_df["binary_prediction"] = False

    slides_df = pd.concat([positive_slides_df, negative_slides_df])
    tiles_df = pd.concat([positive_tiles_df, negative_tiles_df])
    slides_df = slides_df[slides_df.id.isin(tiles_df.slide_id.unique())]

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.dataset.name,
    )


if __name__ == "__main__":
    main()
