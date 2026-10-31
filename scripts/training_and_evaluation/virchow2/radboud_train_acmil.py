from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-train-virchow2-acmil",
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
           experiment/training_and_evaluation/virchow2/data=SL/radboud \
           experiment/training_and_evaluation/virchow2/model=SL/acmil \
           model.weight_decay=... model.lr=... model.attention_dim=... \
           model.mlp_hidden_dim=... model.stkim_k=.. model.stkim_p=... \
           model.loss_p_weight=... model.loss_d_weight=... model.loss_b_weight=... \
        """,
    ],
)
