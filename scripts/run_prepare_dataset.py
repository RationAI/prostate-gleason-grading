from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-prepare-dataset",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="4Gi",
    public=False,
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/pathology/prostate-gleason-grading workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.prepare_dataset +data=mmci2k_224",
    ],
)
