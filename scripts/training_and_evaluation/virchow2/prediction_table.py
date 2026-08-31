from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-prediction-table",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    gpu=...,
    memory="20Gi",
    public=False,
    script=[
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        """uv run -m ml \
           experiment=/training_and_evaluation/virchow2/prediction_table \
           experiment/training_and_evaluation/virchow2/data=mmci2k \
           experiment/training_and_evaluation/virchow2/model=lbfgs \
           model.weight_decay=0 checkpoint=...\
        """,
    ],
)
