from pathlib import Path
from typing import Any

import httpx
import hydra
import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
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
    tiles: pd.DataFrame,
    tiles_output_path: Path,
    encoder: str,
    block_size: int,
    concurrency: int,
    rows_per_file: int,
) -> None:

    with ray.init(num_cpus=10):
        (
            ray.data.from_arrow(pa.Table.from_pandas(tiles, preserve_index=False))
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
            .write_parquet(str(tiles_output_path), min_rows_per_file=rows_per_file)
        )


def filter_processed_tiles(
    tiles: pd.DataFrame,
    processed_tiles: pd.DataFrame,
) -> pd.DataFrame:

    tiles = tiles.merge(
        processed_tiles,
        on=["slide_id", "x", "y"],
        how="left",
        indicator=True,
    )

    return tiles[tiles["_merge"] == "left_only"].drop(columns=["_merge"])


def get_processed_tiles(tiles_path: Path) -> pd.DataFrame | None:

    if not tiles_path.exists() or not any(tiles_path.iterdir()):
        return None

    pa_dataset = ds.dataset(tiles_path, format="parquet")

    processed_tiles = [
        batch.to_pandas()
        for batch in pa_dataset.to_batches(columns=["slide_id", "x", "y"])
    ]

    if processed_tiles:
        return pd.concat(processed_tiles, ignore_index=True).drop_duplicates()


def load_tiling(
    tiling_path: Path, tiles_output_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:

    slides = pd.read_parquet(tiling_path / "slides.parquet")
    tiles = pd.read_parquet(tiling_path / "tiles.parquet")

    processed_tiles = get_processed_tiles(tiles_output_path)
    if processed_tiles is not None:
        tiles = filter_processed_tiles(tiles, processed_tiles)

    tiles = tiles.join(
        slides.set_index("id")[["path", "level", "tile_extent_x", "tile_extent_y"]],
        on="slide_id",
    )

    return slides, tiles


@with_cli_args(["+preprocessing=compute_embeddings"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    tiling_path = Path(mlflow.artifacts.download_artifacts(config.tiling))
    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    slides_output_path = output_path / "slides"
    slides_output_path.mkdir(parents=True, exist_ok=True)

    tiles_output_path = output_path / "tiles"
    tiles_output_path.mkdir(parents=True, exist_ok=True)

    slides, tiles = load_tiling(tiling_path, tiles_output_path)

    slides.to_parquet(slides_output_path / "slides.parquet", index=False)

    if tiles.empty:
        print("All tiles have already been processed.")
    else:
        compute_embeddings(
            tiles,
            tiles_output_path,
            config.encoder,
            config.block_size,
            config.concurrency,
            config.rows_per_file,
        )

    mlflow.log_artifacts(str(output_path), config.dataset.name)


if __name__ == "__main__":
    main()
