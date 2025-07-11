from dacite import DaciteError


class AlgorithmNotCPUCompatibleException(Exception):
    """Exception raised when an CPU-incompatible algorithm tries to run on CPU."""

    pass


class BraTSContainerException(Exception):
    """Exception raised when an algorithm container fails"""

    pass


class AlgorithmConfigException(DaciteError):
    """Exception raised when the algorithm config file has issues."""

    pass


class ZenodoException(Exception):
    """Exception raised when Zenodo is unreachable or returns an error."""

    pass
