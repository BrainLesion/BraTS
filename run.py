from pathlib import Path
from brats.inferer import Inferer
from brats.constants import AlgorithmKeys

inferer = Inferer(algorithm=AlgorithmKeys.BraTS23_biomedmbz)
base = Path("/home/marcel/Projects/helmholtz/brats/workspace/data/single")
inferer.infer_single(
    t1c=base / "t1c.nii.gz",
    t1n=base / "t1n.nii.gz",
    t2f=base / "t2f.nii.gz",
    t2w=base / "t2w.nii.gz",
    output_file="single_out/biomedmbz_seg.nii.gz",
)


# inferer = Inferer(algorithm=AlgorithmKeys.BraTS23_ferreira)
# base = Path("/home/marcel/Projects/helmholtz/brats/workspace/data/single")
# inferer.infer_single(
#     t1c=base / "t1c.nii.gz",
#     t1n=base / "t1n.nii.gz",
#     t2f=base / "t2f.nii.gz",
#     t2w=base / "t2w.nii.gz",
#     output_file="single_out/fer_seg.nii.gz",
# )


# inferer.infer_batch(
#     data_folder="/home/marcel/Projects/helmholtz/brats/workspace/data",
#     output_folder="out",
# )
# import docker
# from rich.progress import Progress

# client = docker.from_env()


# # Show task progress (red for download, green for extract)
# def show_progress(tasks, line, progress: Progress):
#     if line["status"] == "Downloading":
#         task_key = f'[Download {line["id"]}]'
#     elif line["status"] == "Extracting":
#         task_key = f'[Extract  {line["id"]}]'

#     if task_key not in tasks.keys():
#         tasks[task_key] = progress.add_task(
#             f"{task_key}", total=line["progressDetail"]["total"]
#         )
#     else:
#         progress.update(tasks[task_key], completed=line["progressDetail"]["current"])


# def _ensure_image(image: str):
#     if not client.images.list(name=image):
#         print(f"Pulling docker image {image}")
#         tasks = {}
#         with Progress() as progress:
#             resp = client.api.pull(image, stream=True, decode=True)
#             for line in resp:
#                 show_progress(tasks, line, progress)


# _ensure_image("brainles/brats23_biomedmbz:latest")
