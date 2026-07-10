from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-attach-embeddings",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=10,
    memory="32Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.attach_embeddings +experiment=preprocessing/mmci2k_224_attach_virchow2_embeddings",
    ],
)
