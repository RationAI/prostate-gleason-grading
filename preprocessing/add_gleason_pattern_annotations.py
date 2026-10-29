from pathlib import Path
from typing import Any

import hydra
import mlflow
import pandas as pd
import pyarrow as pa
import ray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.tiling import tile_overlay_overlap
from ray.data import Dataset
from ray.data.expressions import col
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry


def get_roi(roi_corners: list[int]) -> BaseGeometry:
    if len(roi_corners) == 4:
        return box(*roi_corners)
    raise ValueError(
        f"RoI should be represented as 4 integer "
        f"coordinates, got {len(roi_corners)} instead."
    )


def add_mask_path(
    tile: dict[str, Any],
    mask_path: Path,
    gleason_pattern: str,
) -> dict[str, Any]:
    mask_path = mask_path / gleason_pattern / f"{tile['stem']}.tiff"
    tile[f"{gleason_pattern}_mask_path"] = str(mask_path)
    return tile


def add_slide_stem(slides_df: pd.DataFrame) -> pd.DataFrame:
    slides_df["stem"] = slides_df["path"].apply(lambda p: Path(p).stem)
    return slides_df


def overlap_to_percentage(
    tile: dict[str, Any],
    gleason_pattern: str,
) -> dict[str, Any]:

    gp_percentage = tile[f"{gleason_pattern}_overlap"].get("255")

    if gp_percentage is None:
        tile[f"{gleason_pattern}_percentage"] = 0.0
    else:
        tile[f"{gleason_pattern}_percentage"] = gp_percentage

    return tile


def add_gleason_pattern_overlap(
    tiles: Dataset,
    masks_path: Path,
    gleason_pattern: str,
    roi: BaseGeometry,
) -> Dataset:
    return (
        tiles.map(
            add_mask_path,
            fn_kwargs={
                "mask_path": masks_path,
                "gleason_pattern": gleason_pattern,
            },
        )
        .with_column(
            f"{gleason_pattern}_overlap",
            tile_overlay_overlap(
                roi=roi,
                overlay_path=col(f"{gleason_pattern}_mask_path"),
                tile_x=col("x"),
                tile_y=col("y"),
                mpp_x=col("mpp_x"),
                mpp_y=col("mpp_y"),
            ),
            num_cpus=1,
            memory=4 * 1024**3,
        )
        .map(
            overlap_to_percentage,
            fn_kwargs={"gleason_pattern": gleason_pattern},
        )
        .drop_columns([f"{gleason_pattern}_mask_path", f"{gleason_pattern}_overlap"])
    )


def add_annotations(
    slides_df: pd.DataFrame,
    gleason_patterns: list[str],
    tiles_path: Path,
    masks_path: Path,
    output_path: Path,
    roi: BaseGeometry,
) -> None:

    tiles_df = pd.read_parquet(tiles_path)
    tiles_df = tiles_df.join(
        slides_df.set_index("id")[["stem", "mpp_x", "mpp_y"]],
        on="slide_id",
    )

    tiles = ray.data.from_arrow(pa.Table.from_pandas(tiles_df, preserve_index=False))
    for gleason_pattern in gleason_patterns:
        tiles = add_gleason_pattern_overlap(tiles, masks_path, gleason_pattern, roi)

    tiles_df = tiles.drop_columns(["stem", "mpp_x", "mpp_y"]).to_pandas()
    tiles_df.to_parquet(output_path / f"{tiles_path.stem}.parquet", index=False)


@with_cli_args(["+preprocessing=add_gleason_pattern_annotations"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    tiling = Path(mlflow.artifacts.download_artifacts(config.tiling))

    output_path = Path(config.output_path)
    tiles_paths: list[Path] = []

    if (tiling / "slides.parquet").exists() and (tiling / "tiles.parquet").exists():
        slides_df = pd.read_parquet(tiling / "slides.parquet")
        tiles_paths = [tiling / "tiles.parquet"]
        slides_output_path, tiles_output_path = output_path, output_path

    elif (tiling / "slides/slides.parquet").exists() and (tiling / "tiles").exists():
        slides_df = pd.read_parquet(tiling / "slides/slides.parquet")
        tiles_paths = list((tiling / "tiles").glob("*.parquet"))
        slides_output_path = output_path / "slides"
        tiles_output_path = output_path / "tiles"

    else:
        raise ValueError("Unexpected structure of the tiling directory.")

    slides_output_path.mkdir(parents=True, exist_ok=True)
    tiles_output_path.mkdir(parents=True, exist_ok=True)

    if not "stem" in slides_df.columns:
        slides_df = add_slide_stem(slides_df)

    slides_df.to_parquet(slides_output_path / "slides.parquet", index=False)

    for tiles_path in tiles_paths:
        add_annotations(
            slides_df,
            config.gleason_patterns,
            tiles_path,
            Path(config.masks_path),
            tiles_output_path,
            get_roi(config.roi),
        )

    mlflow.log_artifacts(str(output_path), config.dataset.name)


if __name__ == "__main__":
    main()
