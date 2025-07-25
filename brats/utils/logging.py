import sys
from typing import Optional, Union

from loguru import logger

# Singleton handler reference
_console_handler_id: Optional[int] = None


def _reset_logging_state_for_tests():
    """
    Reset internal logging state. Only intended for use in tests.
    """
    global _console_handler_id
    logger.remove()
    _console_handler_id = None


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

    Follows the singleton pattern, ensuring only one console handler is present at any time.

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
