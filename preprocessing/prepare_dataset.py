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

    tiling_path = mlflow.artifacts.download_artifacts(
        artifact_uri=config.dataset.mlflow_uris.tiling_with_gs
    )

    slides_df = pd.read_parquet(tiling_path + "/slides.parquet")
    tiles_df = pd.read_parquet(tiling_path + "/tiles.parquet")

    slides_df = slides_df[
        (
            slides_df.gleason_score.isin(config.gleason_scores_to_keep)
            | ~slides_df.carcinoma
        )
        & ~slides_df.stem.isin(config.qc_defective_slides)
    ]

    tiles_df = tiles_df[tiles_df.slide_id.isin(slides_df.id)]

    annotations = pd.read_json(
        mlflow.artifacts.download_artifacts(
            config.dataset.mlflow_uris.cancer_prediction_table
        ),
        orient="split",
    )

    annotations = annotations.join(
        slides_df[slides_df.carcinoma].set_index("stem")["id"],
        on="slide",
        how="inner",
    )

    # embeddings are only accessible by an index - the row order cannot change
    tiles_df["_row_order"] = range(len(tiles_df))

    tiles_df = tiles_df.join(
        annotations.set_index(["id", "x", "y"])["prediction"],
        on=["slide_id", "x", "y"],
        how="left",
    )

    tiles_df = tiles_df.sort_values("_row_order").drop(columns="_row_order")

    tiles_df.prediction = tiles_df.prediction.fillna(0)

    save_mlflow_dataset(
        slides=slides_df,
        tiles=tiles_df,
        dataset_name=config.dataset.name,
    )


if __name__ == "__main__":
    main()
