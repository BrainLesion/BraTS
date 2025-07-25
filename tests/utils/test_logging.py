import pytest
from loguru import logger

from brats.utils.logging import (
    add_console_handler,
    disable,
    enable,
    remove_console_handler,
)
import brats.utils.logging


@pytest.fixture(autouse=True)
def reset_logger_handlers():
    # Reset loguru and internal handler tracking
    logger.remove()
    brats.utils.logging._console_handler_id = None
    yield
    logger.remove()
    brats.utils.logging._console_handler_id = None


def test_disable_and_enable(monkeypatch):
    # monkeypatch internal enabled/disabled registry
    enabled_modules = {}

    def fake_disable(name):
        enabled_modules[name] = False

    def fake_enable(name):
        enabled_modules[name] = True

    monkeypatch.setattr(logger, "disable", fake_disable)
    monkeypatch.setattr(logger, "enable", fake_enable)

    disable()
    assert enabled_modules.get("brats") is False

    enable()
    assert enabled_modules.get("brats") is True


def test_add_console_handler_writes_to_stderr(capfd):
    add_console_handler(level="INFO")

    logger.info("Hello from brats!")
    out, err = capfd.readouterr()

    assert "Hello from brats!" in err
    assert out == ""


def test_add_console_handler_respects_level(capfd):
    add_console_handler(level="ERROR")

    logger.warning("This is a warning")
    logger.error("This is an error")

    out, err = capfd.readouterr()

    assert "This is a warning" not in err
    assert "This is an error" in err
    assert out == ""


def test_add_console_handler_is_singleton(capfd):
    # Add handler the first time
    add_console_handler(level="INFO")
    logger.info("First")

    # Try adding again — shouldn't create a duplicate
    add_console_handler(level="INFO")
    logger.info("Second")

    out, err = capfd.readouterr()
    assert err.count("First") == 1
    assert err.count("Second") == 1


def test_remove_console_handler_stops_logging(capfd):
    add_console_handler(level="INFO")
    logger.info("Will appear")

    remove_console_handler()
    logger.info("Will NOT appear")

    out, err = capfd.readouterr()
    assert "Will appear" in err
    assert "Will NOT appear" not in err


def test_remove_console_handler_idempotent():
    # Should not raise if called without adding first
    remove_console_handler()
    # Can call twice in a row
    remove_console_handler()
