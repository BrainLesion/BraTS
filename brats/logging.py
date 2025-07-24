import sys
from loguru import logger


def disable():
    """
    Disable the logging for the brats package.
    """

    logger.disable("brats")


def enable():
    """
    Enable the logging for the brats package.
    """

    logger.enable("brats")


def add_console_handler(level: str = "WARNING"):
    """
    Add a console handler to the logger for the brats package.
    """
    logger.add(
        sys.stderr,
        level=level,
    )
