from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Dict, Optional

import yaml
from dacite import from_dict


@dataclass
class MetaData:
    """Dataclass for the meta data"""

    authors: str
    """The authors of the algorithm"""
    paper: str
    """If available, a url to the paper of the algorithm"""
    challenge: str
    """The challenge the algorithm was submitted to"""


@dataclass
class RunArgs:
    """Dataclass for the run arguments"""

    docker_image: str
    """The Docker image containing the algorithm"""
    parameters_file: bool
    """Whether the algorithm requires a parameters file"""
    requires_root: bool
    """Whether the Docker container requires root access. This is !discouraged! but some submission do not work without it"""
    shm_size: Optional[str] = "2gb"
    """The required shared memory size for the Docker container"""
    cpu_compatible: Optional[bool] = False
    """Whether the algorithm is compatible with CPU"""


@dataclass
class WeightsData:
    """Dataclass for the weights data"""

    record_id: str
    """The Zenodo record ID of the weights"""
    param_name: Optional[str] = "weights"
    """The parameter that specifies the weights folder in the algorithm execution, typically 'weights' but differs for some"""


@dataclass
class AlgorithmData:
    """Dataclass for the algorithm data"""

    meta: MetaData
    """The meta data of the algorithm"""
    run_args: RunArgs
    """The run arguments of the algorithm"""
    weights: Optional[WeightsData]
    """The weights data of the algorithm. Optional since some algorithms include weights in the docker image"""


@dataclass
class AlgorithmList:
    algorithms: Dict[str, AlgorithmData]


def load_algorithms(file_path: Path) -> Dict[str, AlgorithmData]:
    """Load the algorithms data from the specified yaml file

    Params:
        file_path (str): The path to the yaml file

    Raises:
        FileNotFoundError: If the file is not found

    Returns:
        Dict[str, AlgorithmData]: Dict of algorithm @AlgorithmKeys:@AlgorithmData  pairs
    """
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Algorithm meta data file not found")

    # Convert the dictionary to the dataclass
    return from_dict(data_class=AlgorithmList, data=data).algorithms


def standardize_subject_inputs(
    data_folder: Path,
    subject_id: str,
    t1c: Path | str,
    t1n: Path | str,
    t2f: Path | str,
    t2w: Path | str,
):
    """Standardize the input images for a single subject to match requirements of all algorithms and save them in @data_folder/@subject_id.
        Meaning, e.g. for adult glioma:
            BraTS-GLI-00000-000 \n
            ┣ BraTS-GLI-00000-000-t1c.nii.gz \n
            ┣ BraTS-GLI-00000-000-t1n.nii.gz \n
            ┣ BraTS-GLI-00000-000-t2f.nii.gz \n
            ┗ BraTS-GLI-00000-000-t2w.nii.gz \n

    Args:
        data_folder (Path): Parent folder where the subject folder will be created
        subject_id (str): Subject ID to be used for the folder and filenames
        t1c (Path | str): T1c image path
        t1n (Path | str): T1n image path
        t2f (Path | str): T2f image path
        t2w (Path | str): T2w image path
    """
    subject_folder = data_folder / subject_id
    subject_folder.mkdir(parents=True, exist_ok=True)

    # os.symlink would be more efficient but can cause issues on windows
    # TODO: use symlink on unix systems
    shutil.copy(t1c, subject_folder / f"{subject_id}-t1c.nii.gz")
    shutil.copy(t1n, subject_folder / f"{subject_id}-t1n.nii.gz")
    shutil.copy(t2f, subject_folder / f"{subject_id}-t2f.nii.gz")
    shutil.copy(t2w, subject_folder / f"{subject_id}-t2w.nii.gz")
