from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from rationai.masks.mask_builders import ScalarMaskBuilder
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def assemble_heatmap(
    slide: pd.Series, predictions: pd.DataFrame, save_dir: Path
) -> None:

    mask_builder = ScalarMaskBuilder(
        save_dir=save_dir,
        filename=slide.stem,
        extent_x=slide.extent_x,
        extent_y=slide.extent_y,
        mpp_x=slide.mpp_x,
        mpp_y=slide.mpp_y,
        extent_tile=slide.tile_extent_x,
        stride=slide.stride_x,
    )

    mask_builder.update(
        torch.tensor(predictions["prediction"].values),
        torch.tensor(predictions["x"].values),
        torch.tensor(predictions["y"].values),
    )

    path = mask_builder.save()

    mlflow.log_artifact(str(path), artifact_path="heatmaps")

    path.unlink()


@with_cli_args(["+preprocessing=assemble_heatmaps"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    table = pd.read_json(
        mlflow.artifacts.download_artifacts(
            config.dataset.mlflow_uris.cancer_prediction_table
        ),
        orient="split",
    )

    slides_df = pd.read_parquet(
        mlflow.artifacts.download_artifacts(
            config.dataset.mlflow_uris.tiling_with_gs_filtered + "/slides.parquet"
        )
    )

    tmp_dir = Path("tmp_dir")
    tmp_dir.mkdir(exist_ok=True)

    for _, slide in slides_df.iterrows():
        assemble_heatmap(slide, table[table.slide == slide.stem], tmp_dir)


if __name__ == "__main__":
    main()
