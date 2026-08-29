from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-train-virchow2-adamw",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=10,
    gpu=...,
    memory="40Gi",
    public=False,
    script=[
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        """uv run -m ml \
           experiment=/training_and_evaluation/virchow2/train \
           experiment/training_and_evaluation/virchow2/data=mmci2k \
           experiment/training_and_evaluation/virchow2/model=adamw \
           validation_fold=... model.weight_decay=... model.lr=... \
        """,
    ],
)
