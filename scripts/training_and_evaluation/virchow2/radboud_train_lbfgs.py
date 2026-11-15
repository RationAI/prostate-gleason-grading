from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-train-virchow2-lbfgs",
    image="cerit.io/rationai/base:2.0.6",
    cpu=10,
    gpu="mig-2g.10gb",
    memory="160Gi",
    public=False,
    script=[
        "export MLFLOW_TRACKING_URI=http://mlflow-s3.rationai-mlflow",
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        """uv run -m ml \
           experiment=/training_and_evaluation/virchow2/train \
           experiment/training_and_evaluation/virchow2/data=TL/radboud \
           experiment/training_and_evaluation/virchow2/model=TL/lbfgs \
           model.num_classes=4 \
    	   seed=42 \
           model.weight_decay=0.0,1e-5,1e-4,1e-3,1e-2,0.1,1 \
           	--multirun \
        """,
    ],
)
