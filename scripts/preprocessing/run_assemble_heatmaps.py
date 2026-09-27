from kube_jobs import submit_job


submit_job(
    job_name="prostate-gleason-assemble-heatmaps",
    username=...,
    image="cerit.io/rationai/base:2.0.6",
    cpu=1,
    memory="8Gi",
    public=False,
    script=[
        "git clone https://github.com/RationAI/prostate-gleason-grading.git workdir",
        "cd workdir",
        "uv sync",
        "uv run -m preprocessing.assemble_heatmaps +data=...",
    ],
)
