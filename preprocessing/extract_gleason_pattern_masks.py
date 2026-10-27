from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyvips
import ray
import tifffile
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.masks.processing import process_items
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.masks import write_big_tiff


@ray.remote(num_cpus=1, memory=(5 * 1024**3))
def extract_mask(
    slide: pd.Series,
    masks_path: Path,
    output_path: Path,
    gleason_pattern_codes: dict[str, list[int]],
) -> None:

    slide_name = Path(slide["path"]).stem
    mppx, mppy = slide["mpp_x"], slide["mpp_y"]
    tile_extent_x = slide["tile_extent_x"]
    tile_extent_y = slide["tile_extent_y"]

    mask_path = masks_path / f"{slide_name}_mask.tiff"
    mask: NDArray[np.uint8] = tifffile.imread(mask_path)[..., 0]

    for gleason_pattern, codes in gleason_pattern_codes.items():
        gleason_pattern_mask = np.isin(mask, codes).astype(np.uint8) * 255

        write_big_tiff(
            pyvips.Image.new_from_array(gleason_pattern_mask),
            output_path / gleason_pattern / f"{slide_name}.tiff",
            mppx,
            mppy,
            tile_extent_x,
            tile_extent_y,
        )


@with_cli_args(["+preprocessing=extract_gleason_pattern_masks"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:

    slides = pd.read_parquet(mlflow.artifacts.download_artifacts(config.slides))

    output_path = Path(config.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for gleason_pattern in config.gleason_pattern_codes:
        (output_path / gleason_pattern).mkdir(parents=True, exist_ok=True)

    with ray.init(num_cpus=config.concurrency):
        process_items(
            items=[row for _, row in slides.iterrows()],
            process_item=extract_mask,
            fn_kwargs={
                "masks_path": Path(config.masks_path),
                "output_path": output_path,
                "gleason_pattern_codes": config.gleason_pattern_codes,
            },
            max_concurrent=config.concurrency,
        )

    mlflow.log_artifacts(str(output_path), config.dataset.name)


if __name__ == "__main__":
    main()
