from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-train-virchow2-adamw",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=8,
    gpu=...,
    memory="64Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m ml +experiment=/training/mmci2k/embeddings/virchow2/adamw datamodule.fold=... datamodule.invert_fold_selection=... model.lr=...",
    ],
)
