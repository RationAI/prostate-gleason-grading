import shutil
from pathlib import Path
from typing import Any

import httpx
import hydra
import mlflow
import pandas as pd
import pyarrow as pa
import ray
from omegaconf import DictConfig
from rationai import AsyncClient  # type: ignore[attr-defined]
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling.read_slide_tiles import read_slide_tiles
from ray.data.expressions import col


class Encoder:
    def __init__(self, encoder: str, concurrency: int) -> None:
        self.encoder = encoder
        self.client = AsyncClient(
            limits=httpx.Limits(
                max_connections=concurrency,
                max_keepalive_connections=concurrency,
            ),
            timeout=200,
        )

    async def __call__(self, row: dict[str, Any]) -> dict[str, Any]:
        embedding = (
            (await self.client.models.embed_image(self.encoder, row["tile"]))
            .reshape(-1)
            .tolist()
        )
        del row["tile"]
        row["embedding"] = embedding
        return row


def compute_embeddings(
    encoder: str,
    tiling_path: Path,
    output_path: Path,
    block_size: int,
    concurrency: int,
    rows_per_file: int,
) -> None:

    slides_df = pd.read_parquet(tiling_path / "slides.parquet")

    slides_output_path = output_path / "slides"
    slides_output_path.mkdir(parents=True, exist_ok=True)
    slides_df.to_parquet(slides_output_path / "slides.parquet", index=False)

    tiles_df = pd.read_parquet(tiling_path / "tiles.parquet")
    tiles_df = tiles_df.join(
        slides_df.set_index("id")[["path", "level", "tile_extent_x", "tile_extent_y"]],
        on="slide_id",
    )

    (
        ray.data.from_arrow(pa.Table.from_pandas(tiles_df, preserve_index=False))
        .repartition(target_num_rows_per_block=block_size)
        .with_column(
            "tile",
            read_slide_tiles(
                col("path"),
                col("x"),
                col("y"),
                col("tile_extent_x"),
                col("tile_extent_y"),
                col("level"),
            ),
        )
        .drop_columns(["path", "level", "tile_extent_x", "tile_extent_y"])
        .map(
            Encoder,
            fn_constructor_args=(encoder, concurrency),
            compute=ray.data.ActorPoolStrategy(
                max_size=4, max_tasks_in_flight_per_actor=max(1, concurrency // 4)
            ),
            max_concurrency=concurrency,
        )
        .write_parquet(str(output_path / "tiles"), max_rows_per_file=rows_per_file)
    )


@with_cli_args(["+preprocessing=compute_embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    tiling_path = Path(mlflow.artifacts.download_artifacts(config.tiling))
    output_path = Path(config.output_path)

    if output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    with ray.init(num_cpus=10):
        compute_embeddings(
            config.encoder,
            tiling_path,
            output_path,
            config.block_size,
            config.concurrency,
            config.rows_per_file,
        )

        mlflow.log_artifacts(str(output_path), config.dataset.name)


if __name__ == "__main__":
    main()
