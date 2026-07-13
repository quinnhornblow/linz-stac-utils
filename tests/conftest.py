import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run tests marked as integration",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return

    skip_integration = pytest.mark.skip(
        reason="integration test; use --run-integration to include it"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
