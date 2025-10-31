import logging
from pathlib import Path
from typing import Any, cast

import hydra
import pyvips
import ray
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from openslide import OpenSlide
from rationai.masks import slide_resolution, tissue_mask, write_big_tiff
from rationai.mlkit import autolog
from rationai.mlkit.lightning.loggers import MLFlowLogger

logger = logging.getLogger(__name__)


def process_slide(row: dict[str, Any], output_path: Path) -> dict[str, Any]:
    try:
        with OpenSlide(row["path"]) as slide:
            mpp_x, mpp_y = slide_resolution(slide, level=row["level"])

        slide = cast(
            "pyvips.Image", pyvips.Image.new_from_file(row["path"], level=row["level"])
        )
    except Exception as e:
        logger.error(f"Failed to process slide {row['path']}: {e}", exc_info=True)
        return {"error": e, **row}

    mask = tissue_mask(slide, mpp=(mpp_x + mpp_y) / 2)
    mask_path = output_path / Path(row["path"]).with_suffix(".tiff").name

    write_big_tiff(mask, path=mask_path, mpp_x=mpp_x, mpp_y=mpp_y)
    return {"error": None, **row}


@hydra.main(
    config_path="../configs",
    config_name="preprocessing/tissue_masks",
    version_base=None,
)
@autolog
def main(config: DictConfig, logger: Logger | None = None) -> None:
    assert logger is not None, "Need logger"
    logger = cast("MLFlowLogger", logger)

    tissue_mask_path = Path(config.output_path, config.artifact_paths.tissue_masks)
    tissue_mask_path.mkdir(exist_ok=True, parents=True)
    processing_results_path = Path(
        config.output_path, config.artifact_paths.processing_results
    )

    slides = ray.data.read_csv(config.data_path)
    slides = slides.add_column(
        "level", lambda _: config.level, num_cpus=0.1, memory=128 * 1024**2
    )
    slides = slides.map(
        process_slide,
        fn_kwargs={"output_path": tissue_mask_path},
        num_cpus=1,
        memory=1 * 1024**3,
        max_retries=2,
        retry_exceptions=True,
    )
    slides.write_parquet(str(processing_results_path))

    logger.log_artifacts(
        local_dir=str(tissue_mask_path),
        artifact_path=config.artifact_paths.tissue_masks,
    )
    logger.log_artifacts(
        local_dir=str(processing_results_path),
        artifact_path=config.artifact_paths.processing_results,
    )


if __name__ == "__main__":
    main()
