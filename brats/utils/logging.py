import sys
from typing import Union
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


def add_console_handler(level: Union[str, int] = "WARNING"):
    """
    Add a console handler to the logger for the brats package.

    Args:
        level (str | int): The logging level for the console handler. Defaults to "WARNING".
    """
    logger.add(
        sys.stderr,
        level=level,
    )
