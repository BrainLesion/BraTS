import pytest
from loguru import logger

from brats.utils.logging import add_console_handler, disable, enable


@pytest.fixture(autouse=True)
def reset_logger_handlers():
    # Remove all existing handlers before and after each test
    logger.remove()
    yield
    logger.remove()


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
