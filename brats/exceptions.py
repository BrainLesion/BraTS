from dacite import DaciteError


class AlgorithmNotCPUCompatibleException(Exception):
    """Exception raised when an CPU-incompatible algorithm tries to run on CPU."""

    def __init__(self, message: str):
        super().__init__(message)


class AlgorithmConfigException(DaciteError):
    """Exception raised when the algorithm config file has issues."""

    def __init__(self, message: str):
        super().__init__(message)
