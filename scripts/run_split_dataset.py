from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-split-dataset",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/prostate-gleason-grading.git",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.split_dataset +data=mmci2k_224",
    ],
)
