from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-extract-gleason-pattern-masks",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=6,
    memory="30Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.extract_gleason_pattern_masks data=... masks_path=... slides=...",
    ],
)
