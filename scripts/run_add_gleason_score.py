from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-add-gleason-score",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="8Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/prostate-gleason-grading workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.add_gleason_score +data=mmci2k_512",
    ],
)
