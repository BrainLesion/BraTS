from dacite import DaciteError


class AlgorithmNotCPUCompatibleException(Exception):
    """Exception raised when an CPU-incompatible algorithm tries to run on CPU."""

    pass


class AlgorithmConfigException(DaciteError):
    """Exception raised when the algorithm config file has issues."""

    pass
