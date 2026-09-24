from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-attention-heatmap",
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
        """uv run -m ml \
           experiment=/training_and_evaluation/virchow2/attention_heatmap \
           experiment/training_and_evaluation/virchow2/data/SL=mmci2k \
           experiment/training_and_evaluation/virchow2/model/SL=abmil \
           model.weight_decay=0 model.lr=0 model.attention_dim=... checkpoint=...\
        """,
    ],
)
