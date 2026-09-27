from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-compute-embeddings",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=10,
    memory="32Gi",
    shm="10Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.compute_embeddings data=... preprocessing/encoder=...",
    ],
)
