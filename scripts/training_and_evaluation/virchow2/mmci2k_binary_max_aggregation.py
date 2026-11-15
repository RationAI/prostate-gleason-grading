from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-virchow2-lbfgs-binary-max-aggregation",
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    gpu="mig-2g.10gb",
    memory="24Gi",
    public=False,
    script=[
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        """uv run -m ml \
           experiment=/training_and_evaluation/virchow2/binary_aggregation \
           experiment/training_and_evaluation/virchow2/data/TL=mmci2k \
           experiment/training_and_evaluation/virchow2/model/TL=lbfgs_final \
           model.weight_decay=0.0 \
           'checkpoint=mlflow-artifacts:/73/b7fc4b2270e04b6db524d5f16c1f96de/artifacts/checkpoints/last/checkpoint.ckpt' \
        """,
    ],
)
