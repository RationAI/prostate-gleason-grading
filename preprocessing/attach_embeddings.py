from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
import ray
import torch
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ray.data import SaveMode


def process_group(
    group: pd.DataFrame, embeddings_dir: Path, embeddings_name: str
) -> pd.DataFrame:

    if group["path"].nunique() != 1:
        raise ValueError("Expected one unique path per group")

    slide_path = group["path"].iloc[0]
    slide_name = Path(slide_path).stem

    embeddings = torch.load(
        embeddings_dir / f"{slide_name}.pt",
        weights_only=True,
        map_location="cpu",
    ).numpy()

    if len(group) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(group)} tiles vs {len(embeddings)} embeddings for {slide_name}"
        )

    group = group.copy().sort_values("_row_order").reset_index(drop=True)
    group[f"{embeddings_name}_embedding"] = embeddings.tolist()
    return group


def attach_embeddings(
    tiling_path: Path,
    embeddings_dir: Path,
    output_dir: Path,
    rows_per_file: int,
    embeddings_name: str,
) -> None:

    slides_df = pd.read_parquet(tiling_path / "slides.parquet")
    tiles_df = pd.read_parquet(tiling_path / "tiles.parquet")

    slides_dir = output_dir / "slides"
    tiles_dir = output_dir / "tiles"

    slides_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir.mkdir(parents=True, exist_ok=True)

    slides_df.to_parquet(slides_dir / "slides.parquet", index=False)

    tiles_df = tiles_df.join(slides_df.set_index("id")[["path"]], on="slide_id")
    tiles_df["_row_order"] = range(len(tiles_df))

    (
        ray.data.from_pandas(tiles_df)
        .groupby("slide_id")
        .map_groups(
            process_group,
            fn_kwargs={
                "embeddings_dir": embeddings_dir,
                "embeddings_name": embeddings_name,
            },
            batch_format="pandas",
        )
        .drop_columns(["path", "_row_order"])
        .write_parquet(
            str(tiles_dir), max_rows_per_file=rows_per_file, mode=SaveMode.OVERWRITE
        )
    )


@with_cli_args(["+preprocessing=attach_embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    embeddings_dir = Path(mlflow.artifacts.download_artifacts(config.embeddings_uri))
    tiling_path = Path(
        mlflow.artifacts.download_artifacts(config.dataset.mlflow_uris.tiling_splits)
    )

    with TemporaryDirectory() as tmp_dir, ray.init(num_cpus=10):
        for split in ["test", "train"]:
            attach_embeddings(
                tiling_path / split,
                embeddings_dir,
                Path(tmp_dir) / split,
                config.rows_per_file,
                config.embeddings_name,
            )

        mlflow.log_artifacts(str(tmp_dir), config.dataset.name)


if __name__ == "__main__":
    main()
