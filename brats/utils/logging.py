import sys
from typing import Union
from loguru import logger

# Singleton handler reference
_console_handler_id: int | None = None


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

    Only adds the handler once.

    Args:
        level (str | int): The logging level for the console handler. Defaults to "WARNING".
    """
    global _console_handler_id

    if _console_handler_id is None:
        _console_handler_id = logger.add(sys.stderr, level=level)
    else:
        logger.remove(_console_handler_id)
        _console_handler_id = logger.add(sys.stderr, level=level)


def remove_console_handler():
    """
    Remove the console handler if it was added.
    """
    global _console_handler_id

    if _console_handler_id is not None:
        logger.remove(_console_handler_id)
        _console_handler_id = None
