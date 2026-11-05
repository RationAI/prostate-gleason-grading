from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-acmil-eval",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    gpu=...,
    memory="30Gi",
    public=False,
    script=[
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m ml experiment=/training_and_evaluation/virchow2/acmil_evaluation checkpoint=...",
    ],
)
