from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from dacite import from_dict

PACKAGE_DIR = Path(__file__).parent
META_DATA_FILE = PACKAGE_DIR / "algorithms.yml"


@dataclass
class Author:
    name: str


@dataclass
class AlgorithmData:
    authors: List[Author]
    paper: str
    image: str
    zenodo_record_id: Optional[str]
    shm_size: Optional[str] = "1gb"
    requires_root: Optional[bool] = False
    parameters_file: Optional[bool] = False
    weights_parameter_name: Optional[str] = "weights"


@dataclass
class AlgorithmList:
    algorithms: Dict[str, AlgorithmData]


def load_algorithms() -> Dict[str, AlgorithmData]:
    try:
        with open(META_DATA_FILE, "r") as file:
            data = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError("Algorithm meta data file not found")

    # Wrap the data to fit into the AlgorithmList dataclass
    wrapped_data = {"algorithms": data}

    # Convert the dictionary to the dataclass
    return from_dict(data_class=AlgorithmList, data=wrapped_data).algorithms
