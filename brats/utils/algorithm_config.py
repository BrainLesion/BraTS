from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from brats.utils.exceptions import AlgorithmConfigException

import yaml
from dacite import DaciteError, from_dict


@dataclass
class MetaData:
    """Dataclass for the meta data"""

    authors: str
    """The authors of the algorithm"""
    paper: str
    """If available, a url to the paper of the algorithm"""
    challenge: str
    """The challenge name the algorithm was submitted to"""
    challenge_manuscript: str
    """Arxiv url to the manuscript of the challenge"""
    rank: str
    """The rank of the algorithm in the challenge"""
    year: int
    """The year the algorithm was submitted"""


@dataclass
class RunArgs:
    """Dataclass for the run arguments"""

    docker_image: str
    """The Docker image containing the algorithm"""
    input_name_schema: str
    """The input name schema for the algorithm"""
    parameters_file: bool
    """Whether the algorithm requires a parameters file"""
    requires_root: bool
    """Whether the Docker container requires root access. This is !discouraged! but some submission do not work without it"""
    shm_size: Optional[str] = "2gb"
    """The required shared memory size for the Docker container"""
    cpu_compatible: Optional[bool] = False
    """Whether the algorithm is compatible with CPU"""
    subject_modality_separator: Optional[str] = "-"
    """The separator between the subject ID and the modality, differs e.g. for BraTS24 Meningioma Challenge"""


@dataclass
class AdditionalFilesData:
    """Dataclass for the additional files data"""

    record_id: str
    """The Zenodo record ID of the additional files"""
    param_name: Optional[List[str]] = field(default_factory=lambda: ["weights"])
    """The parameter(s) that specify additional file(s) in the algorithm execution, typically 'weights' but differs for some and can be multiple"""
    param_path: Optional[List[str]] = None
    """The path(s) to specific file(s) / folder(s) in the additional files folder. Not required since some algorithms accept the entire additional files folder"""


@dataclass
class AlgorithmData:
    """Dataclass for the algorithm data"""

    meta: MetaData
    """The meta data of the algorithm"""
    run_args: RunArgs
    """The run arguments of the algorithm"""
    additional_files: Optional[AdditionalFilesData]
    """The additional files of the algorithm. Optional since some algorithms include them in the docker image"""


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

    try:
        # Convert the dictionary to the dataclass
        algorithms = from_dict(data_class=AlgorithmList, data=data).algorithms
    except DaciteError as e:
        raise AlgorithmConfigException(f"Error loading algorithm data: {e}")
    return algorithms
