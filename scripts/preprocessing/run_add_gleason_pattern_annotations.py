from kube_jobs import storage, submit_job


submit_job(
    job_name="prostate-gleason-add-gp-annotations",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=4,
    memory="20Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run python -m preprocessing.add_gleason_pattern_annotations data=... tiling_path=... masks_path=...",
    ],
    storage=[storage.secure.DATA, storage.secure.PROJECTS],
)
