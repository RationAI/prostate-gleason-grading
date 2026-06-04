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

    assert group["path"].nunique() == 1, "Expected one unique path per group"

    slide_path = group["path"].iloc[0]
    slide_name = Path(slide_path).stem

    embeddings = (
        torch.load(
            str(embeddings_dir / f"{slide_name}.pt"),
            map_location="cpu",
        )
        .cpu()
        .numpy()
    )

    if len(group) != len(embeddings):
        raise ValueError(
            f"Mismatch: {len(group)} tiles vs {len(embeddings)} embeddings for {slide_name}"
        )

    group = group.copy()
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

    slides_dir.mkdir(parents=True)
    tiles_dir.mkdir(parents=True)

    slides_df.to_parquet(slides_dir / "slides.parquet", index=False)

    with ray.init(num_cpus=10):
        tiles_df = tiles_df.join(slides_df.set_index("id")[["path"]], on="slide_id")

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
            .drop_columns(["path"])
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

    with TemporaryDirectory() as tmp_dir:
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
