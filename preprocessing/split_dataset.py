import hydra
import mlflow
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.tiling.writers import save_mlflow_dataset
from ratiopath.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold


def stratified_group_k_fold_split(
    slides_df: pd.DataFrame,
    target_col: str,
    group_col: str,
    k: int,
    random_state: int,
) -> pd.DataFrame:

    slides_df = slides_df.copy()
    fold_mask = np.empty(len(slides_df), dtype=int)

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = sgkf.split(slides_df, slides_df[target_col], slides_df[group_col])

    for i, (_, fold_index) in enumerate(folds):
        fold_mask[fold_index] = i

    slides_df["fold"] = fold_mask

    return slides_df


@with_cli_args(["+preprocessing=split_dataset"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    annotations_source = pd.read_csv(
        mlflow.artifacts.download_artifacts(
            config.dataset.mlflow_uris.annotations_source
        )
    )

    tiling_uri = config.dataset.mlflow_uris.tiling_done
    slides_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/slides.parquet")
    )
    tiles_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(tiling_uri + "/tiles.parquet")
    )

    # np.unique() called from ratiopath.model_selection.train_test_split
    # cannot handle arrays containing both strings and None values
    slides_df.gleason_score = slides_df.gleason_score.fillna("None")

    slides_df = slides_df.join(
        annotations_source.set_index("slide_path")[config.group_column],
        on="path",
        how="left",
    )

    if slides_df[config.group_column].isna().any():
        raise ValueError(
            f"Missing '{config.group_column}' after joining annotations source."
        )

    train, test = train_test_split(
        slides_df,
        stratify=slides_df.gleason_score,
        groups=slides_df[config.group_column],
        test_size=config.test_size / 100,
        random_state=config.random_state,
    )

    train = stratified_group_k_fold_split(
        train,
        "gleason_score",
        config.group_column,
        config.fold_size,
        config.random_state,
    )

    save_mlflow_dataset(
        slides=test.drop(config.group_column, axis=1),
        tiles=tiles_df[tiles_df.slide_id.isin(test.id)],
        dataset_name=config.dataset.name + "/test",
    )

    save_mlflow_dataset(
        slides=train.drop(config.group_column, axis=1),
        tiles=tiles_df[tiles_df.slide_id.isin(train.id)],
        dataset_name=config.dataset.name + "/train",
    )


if __name__ == "__main__":
    main()
